from __future__ import print_function, division, absolute_import
from pprint import pprint
from contextlib import contextmanager
from collections import namedtuple, defaultdict
import warnings
import inspect

from numba import (bytecode, interpreter, typing, typeinfer, lowering,
                   objmode, irpasses, utils, config, type_annotations,
                   types, ir, assume, looplifting, macro, types)
from numba.targets import cpu


class Flags(utils.ConfigOptions):
    # These options are all false by default, but the defaults are
    # different with the @jit decorator (see targets.options.TargetOptions).

    OPTIONS = frozenset([
        # Enable loop-lifting
        'enable_looplift',
        # Enable pyobject mode (in general)
        'enable_pyobject',
        # Enable pyobject mode inside lifted loops
        'enable_pyobject_looplift',
        # Force pyobject mode inside the whole function
        'force_pyobject',
        # Release GIL inside the native function
        'release_gil',
        'no_compile',
        'boundcheck',
        'forceinline',
        'no_cpython_wrapper',
    ])


DEFAULT_FLAGS = Flags()


CR_FIELDS = ["typing_context",
             "target_context",
             "entry_point",
             "typing_error",
             "type_annotation",
             "signature",
             "objectmode",
             "lifted",
             "fndesc",
             "interpmode",
             "library",
             "exception_map",
             "environment"]


CompileResult = namedtuple("CompileResult", CR_FIELDS)
FunctionAttributes = namedtuple("FunctionAttributes",
                                ['name', 'filename', 'lineno'])
DEFAULT_FUNCTION_ATTRIBUTES = FunctionAttributes('<anonymous>', '<unknown>', 0)


_LowerResult = namedtuple("_LowerResult", [
    "fndesc",
    "exception_map",
    "cfunc",
    "env",
])


def get_function_attributes(func):
    '''
    Extract the function attributes from a Python function or object with
    *py_func* attribute, such as CPUOverloaded.

    Returns an instance of FunctionAttributes.
    '''
    if hasattr(func, 'py_func'):
        func = func.py_func  # This is a Overload object

    name, filename, lineno = DEFAULT_FUNCTION_ATTRIBUTES
    try:
        name = func.__name__
    except AttributeError:
        pass  # this "function" object isn't really a function

    try:
        possible_filename = inspect.getsourcefile(func)
        # Sometimes getsourcefile returns null
        if possible_filename is not None:
            filename = possible_filename
    except TypeError:
        pass  # built-in function, or other object unsupported by inspect

    try:
        lines, lineno = inspect.getsourcelines(func)
    except (IOError, TypeError):
        pass  # unable to read source code for function

    return FunctionAttributes(name, filename, lineno)


def compile_result(**kws):
    keys = set(kws.keys())
    fieldset = set(CR_FIELDS)
    badnames = keys - fieldset
    if badnames:
        raise NameError(*badnames)
    missing = fieldset - keys
    for k in missing:
        kws[k] = None
    return CompileResult(**kws)


def compile_isolated(func, args, return_type=None, flags=DEFAULT_FLAGS,
                     locals={}):
    """
    Compile the function is an isolated environment.
    Good for testing.
    """
    typingctx = typing.Context()
    targetctx = cpu.CPUContext(typingctx)
    return compile_extra(typingctx, targetctx, func, args, return_type, flags,
                         locals)


class _CompileStatus(object):
    """
    Used like a C record
    """
    __slots__ = ['fail_reason', 'can_fallback', 'can_giveup']

    def __init__(self, can_fallback, can_giveup):
        self.fail_reason = None
        self.can_fallback = can_fallback
        self.can_giveup = can_giveup

    def __repr__(self):
        vals = []
        for k in self.__slots__:
            vals.append("{k}={v}".format(k=k, v=getattr(self, k)))
        return ', '.join(vals)


class _EarlyPipelineCompletion(Exception):
    def __init__(self, result):
        self.result = result


class _PipelineManager(object):
    def __init__(self):
        self.pipeline_order = []
        self.pipeline_stages = {}
        self._finalized = False

    def create_pipeline(self, pipeline_name):
        assert not self._finalized, "Pipelines can no longer be added"
        self.pipeline_order.append(pipeline_name)
        self.pipeline_stages[pipeline_name] = []
        self.current = pipeline_name

    def add_stage(self, stage_function, stage_description):
        assert not self._finalized, "Stages can no longer be added."
        current_pipeline_name = self.pipeline_order[-1]
        func_desc_tuple = (stage_function, stage_description)
        self.pipeline_stages[current_pipeline_name].append(func_desc_tuple)

    def finalize(self):
        self._finalized = True

    def _patch_error(self, desc, exc):
        """
        Patches the error to show the stage that it arose in.
        """
        newmsg = "{desc}\n{exc}".format(desc=desc, exc=exc)
        exc.args = (newmsg,)
        return exc

    def run(self, status):
        assert self._finalized, "PM must be finalized before run()"
        res = None
        for pipeline_name in self.pipeline_order:
            is_final_pipeline = pipeline_name == self.pipeline_order[-1]
            for stage, stage_name in self.pipeline_stages[pipeline_name]:
                try:
                    res = stage()
                except _EarlyPipelineCompletion as e:
                    return e.result
                except BaseException as e:
                    msg = "Failed at %s (%s)" % (pipeline_name, stage_name)
                    patched_exception = self._patch_error(msg, e)
                    # No more fallback pipelines?
                    if is_final_pipeline:
                        raise patched_exception
                    # Go to next fallback pipeline
                    else:
                        status.fail_reason = patched_exception
                        break
            else:
                return res

        # TODO save all error information
        raise CompilerError("All pipelines have failed")


class CompilerError(Exception):
    pass


class Pipeline(object):
    """
    Stores and manages states for the compiler pipeline
    """
    def __init__(self, typingctx, targetctx, library, args, return_type, flags,
                 locals):
        self.typingctx = typingctx
        self.targetctx = targetctx
        self.library = library
        self.args = args
        self.return_type = return_type
        self.flags = flags
        self.locals = locals
        self.bc = None
        self.func_attr = None
        self.lifted = None

        self.status = _CompileStatus(
            can_fallback=self.flags.enable_pyobject,
            can_giveup=config.COMPATIBILITY_MODE
        )

    @contextmanager
    def fallback_context(self, msg):
        """
        Wraps code that would signal a fallback to object mode
        """
        try:
            yield
        except BaseException as e:
            if not self.status.can_fallback:
                raise
            else:
                if utils.PYVERSION >= (3,):
                    # Clear all references attached to the traceback
                    e = e.with_traceback(None)
                warnings.warn_explicit('%s: %s' % (msg, e),
                                       config.NumbaWarning,
                                       self.func_attr.filename,
                                       self.func_attr.lineno)

                raise

    @contextmanager
    def giveup_context(self, msg):
        """
        Wraps code that would signal a fallback to interpreter mode
        """
        try:
            yield
        except BaseException as e:
            if not self.status.can_giveup:
                raise
            else:
                if utils.PYVERSION >= (3,):
                    # Clear all references attached to the traceback
                    e = e.with_traceback(None)
                warnings.warn_explicit('%s: %s' % (msg, e),
                                       config.NumbaWarning,
                                       self.func_attr.filename,
                                       self.func_attr.lineno)

                raise

    def extract_bytecode(self, func):
        """
        Extract bytecode from function
        """
        func_attr = get_function_attributes(func)
        self.func = func
        self.func_attr = func_attr
        bc = bytecode.ByteCode(func=self.func)
        if config.DUMP_BYTECODE:
            print(bc.dump())

        return bc

    def compile_extra(self, func):
        try:
            bc = self.extract_bytecode(func)
        except BaseException as e:
            if self.status.can_giveup:
                return self.stage_compile_interp_mode()
            else:
                raise e

        return self.compile_bytecode(bc, func_attr=self.func_attr)

    def compile_bytecode(self, bc, lifted=(),
                         func_attr=DEFAULT_FUNCTION_ATTRIBUTES):
        self.bc = bc
        self.func = bc.func
        self.lifted = lifted
        self.func_attr = func_attr
        return self._compile_bytecode()

    def compile_internal(self, bc, func_attr=DEFAULT_FUNCTION_ATTRIBUTES):
        assert not self.flags.force_pyobject
        self.bc = bc
        self.lifted = ()
        self.func_attr = func_attr
        self.status.can_fallback = False
        self.status.can_giveup = False
        return self._compile_bytecode()

    def stage_analyze_bytecode(self):
        """
        Analyze bytecode and translating to Numba IR
        """
        self.interp = translate_stage(self.bc)
        self.nargs = len(self.interp.argspec.args)
        if not self.args and self.flags.force_pyobject:
            # Allow an empty argument types specification when object mode
            # is explicitly requested.
            self.args = (types.pyobject,) * self.nargs
        elif len(self.args) != self.nargs:
            raise TypeError("Signature mismatch: %d argument types given, "
                            "but function takes %d arguments"
                            % (len(self.args), self.nargs))

    def frontend_looplift(self):
        """
        Loop lifting analysis and transformation
        """
        assert not self.lifted

        # Try loop lifting
        loop_flags = self.flags.copy()
        outer_flags = self.flags.copy()
        # Do not recursively loop lift
        outer_flags.unset('enable_looplift')
        loop_flags.unset('enable_looplift')
        if not self.flags.enable_pyobject_looplift:
            loop_flags.unset('enable_pyobject')

        def dispatcher_factory(loopbc):
            from . import dispatcher
            return dispatcher.LiftedLoop(loopbc, self.typingctx,
                                         self.targetctx,
                                         self.locals, loop_flags)

        entry, loops = looplifting.lift_loop(self.bc, dispatcher_factory)
        if loops:
            # Some loops were extracted
            if config.DEBUG_FRONTEND or config.DEBUG:
                print("Lifting loop", loops[0].get_source_location())

            cres = compile_bytecode(self.typingctx, self.targetctx, entry,
                                    self.args, self.return_type,
                                    outer_flags, self.locals,
                                    lifted=tuple(loops),
                                    func_attr=self.func_attr)
            return cres

    def stage_objectmode_frontend(self):
        """
        Front-end: Analyze bytecode, generate Numba IR, infer types
        """
        if self.flags.enable_looplift:
            assert not self.lifted
            cres = self.frontend_looplift()
            if cres is not None:
                raise _EarlyPipelineCompletion(cres)

        # Fallback typing: everything is a python object
        self.typemap = defaultdict(lambda: types.pyobject)
        self.calltypes = defaultdict(lambda: types.pyobject)
        self.return_type = types.pyobject

    def stage_nopython_frontend(self):
        """
        Type inference and legalization
        """
        with self.fallback_context('Function "%s" failed type inference'
                                   % (self.func_attr.name,)):
            legalize_given_types(self.args, self.return_type)
            # Type inference
            self.typemap, self.return_type, self.calltypes = type_inference_stage(
                self.typingctx,
                self.interp,
                self.args,
                self.return_type,
                self.locals)

        with self.fallback_context('Function "%s" has invalid return type'
                                   % (self.func_attr.name,)):
            legalize_return_type(self.return_type, self.interp,
                                 self.targetctx)

    def stage_annotate_type(self):
        """
        Create type annotation after type inference
        """
        self.type_annotation = type_annotations.TypeAnnotation(
            interp=self.interp,
            typemap=self.typemap,
            calltypes=self.calltypes,
            lifted=self.lifted)

        if config.ANNOTATE:
            print("ANNOTATION".center(80, '-'))
            print(self.type_annotation)
            print('=' * 80)

    def backend_object_mode(self):
        """
        Object mode compilation
        """
        with self.giveup_context("Function %s failed at object mode lowering"
                                 % (self.func_attr.name,)):
            if len(self.args) != self.nargs:
                # append missing
                self.args = (tuple(self.args) + (types.pyobject,) *
                             (self.nargs - len(self.args)))

            return py_lowering_stage(self.targetctx,
                                     self.library,
                                     self.interp,
                                     self.flags)

    def backend_nopython_mode(self):
        """Native mode compilation"""
        with self.fallback_context("Function %s failed at nopython "
                                   "mode lowering" % (self.func_attr.name,)):
            return native_lowering_stage(
                self.targetctx,
                self.library,
                self.interp,
                self.typemap,
                self.return_type,
                self.calltypes,
                self.flags)

    def _backend(self, lowerfn, objectmode):
        """
        Back-end: Generate LLVM IR from Numba IR, compile to machine code
        """
        if self.library is None:
            codegen = self.targetctx.jit_codegen()
            self.library = codegen.create_library(self.bc.func_qualname)
        lowered = lowerfn()
        signature = typing.signature(self.return_type, *self.args)
        cr = compile_result(typing_context=self.typingctx,
                            target_context=self.targetctx,
                            entry_point=lowered.cfunc,
                            typing_error=self.status.fail_reason,
                            type_annotation=self.type_annotation,
                            library=self.library,
                            exception_map=lowered.exception_map,
                            signature=signature,
                            objectmode=objectmode,
                            interpmode=False,
                            lifted=self.lifted,
                            fndesc=lowered.fndesc,
                            environment=lowered.env,)
        return cr

    def stage_objectmode_backend(self):
        """
        Lowering for object mode
        """
        lowerfn = self.backend_object_mode
        res = self._backend(lowerfn, objectmode=True)

        # Warn if compiled function in object mode and force_pyobject not set
        if not self.flags.force_pyobject:
            if len(self.lifted) > 0:
                warn_msg = 'Function "%s" was compiled in object mode without forceobj=True, but has lifted loops.' % (self.func_attr.name,)
            else:
                warn_msg = 'Function "%s" was compiled in object mode without forceobj=True.' % (self.func_attr.name,)
            warnings.warn_explicit(warn_msg, config.NumbaWarning,
                                   self.func_attr.filename,
                                   self.func_attr.lineno)
            if self.flags.release_gil:
                warn_msg = "Code running in object mode won't allow parallel execution despite nogil=True."
                warnings.warn_explicit(warn_msg, config.NumbaWarning,
                                       self.func_attr.filename,
                                       self.func_attr.lineno)
        return res

    def stage_nopython_backend(self):
        """
        Do lowering for nopython
        """
        lowerfn = self.backend_nopython_mode
        return self._backend(lowerfn, objectmode=False)

    def stage_compile_interp_mode(self):
        """
        Just create a compile result for interpreter mode
        """
        args = [types.pyobject] * len(self.args)
        signature = typing.signature(types.pyobject, *args)
        cr = compile_result(typing_context=self.typingctx,
                            target_context=self.targetctx,
                            entry_point=self.func,
                            typing_error=self.status.fail_reason,
                            type_annotation="<Interpreter mode function>",
                            signature=signature,
                            objectmode=False,
                            interpmode=True,
                            lifted=(),
                            fndesc=None,)
        return cr

    def _compile_bytecode(self):
        pm = _PipelineManager()

        if not self.flags.force_pyobject:
            pm.create_pipeline("nopython")
            pm.add_stage(self.stage_analyze_bytecode, "analyzing bytecode")
            pm.add_stage(self.stage_nopython_frontend, "nopython frontend")
            pm.add_stage(self.stage_annotate_type, "annotate type")
            pm.add_stage(self.stage_nopython_backend, "nopython mode backend")

        if self.status.can_fallback or self.flags.force_pyobject:
            pm.create_pipeline("object")
            pm.add_stage(self.stage_analyze_bytecode, "analyzing bytecode")
            pm.add_stage(self.stage_objectmode_frontend, "object mode frontend")
            pm.add_stage(self.stage_annotate_type, "annotate type")
            pm.add_stage(self.stage_objectmode_backend, "object mode backend")

        if self.status.can_giveup:
            pm.create_pipeline("interp")
            pm.add_stage(self.stage_compile_interp_mode, "compiling with interpreter mode")

        pm.finalize()
        return pm.run(self.status)


def compile_extra(typingctx, targetctx, func, args, return_type, flags,
                  locals, library=None):
    """
    Args
    ----
    - return_type
        Use ``None`` to indicate
    """
    pipeline = Pipeline(typingctx, targetctx, library,
                        args, return_type, flags, locals)
    return pipeline.compile_extra(func)


def compile_bytecode(typingctx, targetctx, bc, args, return_type, flags,
                     locals, lifted=(),
                     func_attr=DEFAULT_FUNCTION_ATTRIBUTES, library=None):

    pipeline = Pipeline(typingctx, targetctx, library,
                        args, return_type, flags, locals)
    return pipeline.compile_bytecode(bc=bc, lifted=lifted, func_attr=func_attr)


def compile_internal(typingctx, targetctx, library,
                     func, args, return_type, flags, locals):
    # For now this is the same thing as compile_extra().
    pipeline = Pipeline(typingctx, targetctx, library,
                        args, return_type, flags, locals)
    return pipeline.compile_extra(func)


def _is_nopython_types(t):
    return not isinstance(t, types.Dummy) or isinstance(t, types.Opaque)


def legalize_given_types(args, return_type):
    # Filter argument types
    for i, a in enumerate(args):
        if not _is_nopython_types(a):
            raise TypeError("Arg %d of %s is not legal in nopython "
                            "mode" % (i, a))
    # Filter return type
    if (return_type and return_type != types.none and
            not _is_nopython_types(return_type)):
        raise TypeError('Return type of %s is not legal in nopython '
                        'mode' % (return_type,))


def legalize_return_type(return_type, interp, targetctx):
    """
    Only accept array return type iff it is passed into the function.
    Reject function object return types if in nopython mode.
    """
    assert assume.return_argument_array_only

    if isinstance(return_type, types.Array):
        # Walk IR to discover all return statements
        retstmts = []
        caststmts = {}
        for bid, blk in interp.blocks.items():
            for inst in blk.body:
                if isinstance(inst, ir.Return):
                    retstmts.append(inst.value.name)
                if (isinstance(inst, ir.Assign)
                        and isinstance(inst.value, ir.Expr)
                        and inst.value.op == 'cast'):
                    caststmts[inst.target.name] = inst.value

        assert retstmts, "No return statements?"

        # FIXME: In the future, we can return an array that is either a dynamically
        #        allocated array or an array that is passed as argument.  This
        #        must be statically resolvable.

        # The return value must be the first modification of the value.
        arguments = set("{0}.1".format(a) for a in interp.argspec.args)

        for var in retstmts:
            cast = caststmts.get(var)
            if cast is None or cast.value.name not in arguments:
                raise TypeError("Only accept returning of array passed into the "
                                "function as argument")

    elif (isinstance(return_type, types.Function) or
            isinstance(return_type, types.Phantom)):
        raise TypeError("Can't return function object in nopython mode")


def translate_stage(bytecode):
    interp = interpreter.Interpreter(bytecode=bytecode)

    if config.DUMP_CFG:
        interp.cfa.dump()

    interp.interpret()

    if config.DEBUG or config.DUMP_IR:
        print(("IR DUMP: %s" % interp.bytecode.func_qualname).center(80, "-"))
        interp.dump()

    expanded = macro.expand_macros(interp.blocks)

    if config.DUMP_IR and expanded:
        print(("MACRO-EXPANDED IR DUMP: %s" % interp.bytecode.func_qualname)
              .center(80, "-"))
        interp.dump()

    return interp


def type_inference_stage(typingctx, interp, args, return_type, locals={}):
    if len(args) != len(interp.argspec.args):
        raise TypeError("Mismatch number of argument types")

    infer = typeinfer.TypeInferer(typingctx, interp.blocks)

    # Seed argument types
    for arg, ty in zip(interp.argspec.args, args):
        infer.seed_type(arg, ty)

    # Seed return type
    if return_type is not None:
        infer.seed_return(return_type)

    # Seed local types
    for k, v in locals.items():
        infer.seed_type(k, v)

    infer.build_constrain()
    infer.propagate()
    typemap, restype, calltypes = infer.unify()

    if config.DEBUG:
        pprint(typemap)
        pprint(restype)
        pprint(calltypes)

    return typemap, restype, calltypes


def native_lowering_stage(targetctx, library, interp, typemap, restype,
                          calltypes, flags):
    # Lowering
    fndesc = lowering.PythonFunctionDescriptor.from_specialized_function(
        interp, typemap, restype, calltypes, mangler=targetctx.mangler,
        inline=flags.forceinline)

    lower = lowering.Lower(targetctx, library, fndesc, interp)
    lower.lower()
    if not flags.no_cpython_wrapper:
        lower.create_cpython_wrapper(flags.release_gil)
    env = lower.env
    exception_map = lower.exceptions
    del lower

    if flags.no_compile:
        return _LowerResult(fndesc, exception_map, cfunc=None, env=None)
    else:
        # Prepare for execution
        cfunc = targetctx.get_executable(library, fndesc, env)
        # Insert native function for use by other jitted-functions.
        # We also register its library to allow for inlining.
        targetctx.insert_user_function(cfunc, fndesc, [library])
        return _LowerResult(fndesc, exception_map, cfunc=cfunc, env=None)


def py_lowering_stage(targetctx, library, interp, flags):
    fndesc = lowering.PythonFunctionDescriptor.from_object_mode_function(interp)
    lower = objmode.PyLower(targetctx, library, fndesc, interp)
    lower.lower()
    if not flags.no_cpython_wrapper:
        lower.create_cpython_wrapper()
    env = lower.env
    exception_map = lower.exceptions
    del lower

    if flags.no_compile:
        return _LowerResult(fndesc, exception_map, cfunc=None, env=env)
    else:
        # Prepare for execution
        cfunc = targetctx.get_executable(library, fndesc, env)
        return _LowerResult(fndesc, exception_map, cfunc, env)


def ir_optimize_for_py_stage(interp):
    """
    This passes breaks semantic for the type inferer but they reduces
    refct calls for object mode.
    """
    irpasses.RemoveRedundantAssign(interp).run()
    if config.DEBUG:
        print("ir optimize".center(80, '-'))
        interp.dump()
