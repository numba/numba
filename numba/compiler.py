from __future__ import print_function, division, absolute_import
from pprint import pprint
from contextlib import contextmanager
from collections import namedtuple, defaultdict
import warnings
import inspect

from numba import (bytecode, interpreter, typing, typeinfer, lowering,
                   objmode, irpasses, utils, config, type_annotations,
                   types, ir, assume, looplifting, macro)
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
        'no_compile',
        'no_wraparound',
        'boundcheck',
        ])


DEFAULT_FLAGS = Flags()


CR_FIELDS = ["typing_context",
             "target_context",
             "entry_point",
             "typing_error",
             "type_annotation",
             "llvm_module",
             "llvm_func",
             "signature",
             "objectmode",
             "lifted",
             "fndesc"]


CompileResult = namedtuple("CompileResult", CR_FIELDS)
FunctionAttributes = namedtuple("FunctionAttributes",
    ['name', 'filename', 'lineno'])
DEFAULT_FUNCTION_ATTRIBUTES = FunctionAttributes('<anonymous>', '<unknown>', 0)


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
    __slots__ = ['fail_reason', 'use_python_mode',
                 'use_interpreter_mode', 'can_fallback', 'can_giveup']


@contextmanager
def _fallback_context(status):
    """Wraps code that would signal a fallback to object mode
    """
    try:
        yield
    except BaseException as e:
        if not status.can_fallback:
            raise
        if utils.PYVERSION >= (3,):
            # Clear all references attached to the traceback
            e = e.with_traceback(None)
        status.fail_reason = e
        status.use_python_mode = True


@contextmanager
def _giveup_context(status):
    """Wraps code that would signal a fallback to interpreter mode
    """
    try:
        yield
    except BaseException as e:
        if not status.can_giveup:
            raise
        if utils.PYVERSION >= (3,):
            # Clear all references attached to the traceback
            e = e.with_traceback(None)
        status.fail_reason = e
        status.use_interpreter_mode = True


class Pipeline(object):
    """Stores and manages states for the compiler pipeline
    """
    def __init__(self, typingctx, targetctx, args, return_type, flags, locals):
        self.typingctx = typingctx
        self.targetctx = targetctx
        self.args = args
        self.return_type = return_type
        self.flags = flags
        self.locals = locals
        self.bc = None
        self.func_attr = None
        self.lifted = None

    def compile_extra(self, func):
        bc = bytecode.ByteCode(func=func)
        if config.DUMP_BYTECODE:
            print(bc.dump())
        func_attr = get_function_attributes(func)
        return self.compile_bytecode(bc, func_attr=func_attr)

    def compile_bytecode(self, bc, lifted=(),
                         func_attr=DEFAULT_FUNCTION_ATTRIBUTES):
        self.bc = bc
        self.lifted = lifted
        self.func_attr = func_attr

        self.status = _CompileStatus()
        self.status.can_fallback = self.flags.enable_pyobject
        self.status.fail_reason = None
        self.status.use_python_mode = self.flags.force_pyobject

        self.targetctx = self.targetctx.localized()
        self.targetctx.metadata['wraparound'] = not self.flags.no_wraparound

        return self._compile_bytecode()

    def analyze_bytecode(self):
        self.interp = translate_stage(self.bc)
        self.nargs = len(self.interp.argspec.args)
        if len(self.args) > self.nargs:
            raise TypeError("Too many argument types")

    def frontend_object_mode(self):
        self.analyze_bytecode()

    def frontend_nopython_mode(self):
        self.analyze_bytecode()
        with _fallback_context(self.status):
            legalize_given_types(self.args, self.return_type)
            # Type inference
            self.typemap, self.return_type, self.calltypes = type_inference_stage(
                self.typingctx,
                self.interp,
                self.args,
                self.return_type,
                self.locals)

        if self.status.fail_reason is not None:
            warnings.warn_explicit('Function "%s" failed type inference: %s'
                                   % (self.func_attr.name,
                                      self.status.fail_reason),
                                   config.NumbaWarning,
                                   self.func_attr.filename,
                                   self.func_attr.lineno)

        if not self.status.use_python_mode:
            with _fallback_context(self.status):
                legalize_return_type(self.return_type, self.interp,
                                     self.targetctx)
            if self.status.fail_reason is not None:
                warnings.warn_explicit('Function "%s" has invalid return type: %s'
                                       % (self.func_attr.name,
                                          self.status.fail_reason),
                                       config.NumbaWarning,
                                       self.func_attr.filename,
                                       self.func_attr.lineno)

    def frontend_looplift(self):
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
            cres = compile_bytecode(self.typingctx, self.targetctx, entry,
                                    self.args, self.return_type,
                                    outer_flags, self.locals,
                                    lifted=tuple(loops),
                                    func_attr=self.func_attr)
            return cres

    def frontend(self):
        """
        Front-end: Analyze bytecode, generate Numba IR, infer types
        """
        if self.status.use_python_mode:
            self.frontend_object_mode()
        else:
            self.frontend_nopython_mode()

        if self.status.use_python_mode and self.flags.enable_looplift:
            assert not self.lifted
            cres = self.frontend_looplift()
            if cres is not None:
                return cres

        if self.status.use_python_mode:
            # Fallback typing: everything is a python object
            self.typemap = defaultdict(lambda: types.pyobject)
            self.calltypes = defaultdict(lambda: types.pyobject)
            self.return_type = types.pyobject

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
        """Object mode compilation"""
        func, lmod, lfunc, fndesc = py_lowering_stage(self.targetctx,
                                                      self.interp,
                                                      self.flags.no_compile)
        if len(self.args) != self.nargs:
            # append missing
            self.args = (tuple(self.args) + (types.pyobject,) *
                    (self.nargs - len(self.args)))

        return func, lmod, lfunc, fndesc

    def backend_nopython_mode(self):
        """Native mode compilation"""
        func, lmod, lfunc, fndesc = native_lowering_stage(self.targetctx,
                                                          self.interp,
                                                          self.typemap,
                                                          self.return_type,
                                                          self.calltypes,
                                                          self.flags.no_compile)
        return func, lmod, lfunc, fndesc

    def backend(self):
        """
        Back-end: Generate LLVM IR from Numba IR, compile to machine code
        """

        func, lmod, lfunc, fndesc = (self.backend_object_mode()
                                     if self.status.use_python_mode
                                     else self.backend_nopython_mode())

        signature = typing.signature(self.return_type, *self.args)

        assert lfunc.module is lmod
        cr = compile_result(typing_context=self.typingctx,
                            target_context=self.targetctx,
                            entry_point=func,
                            typing_error=self.status.fail_reason,
                            type_annotation=self.type_annotation,
                            llvm_func=lfunc,
                            llvm_module=lmod,
                            signature=signature,
                            objectmode=self.status.use_python_mode,
                            lifted=self.lifted,
                            fndesc=fndesc,)

        # Warn if compiled function in object mode and force_pyobject not set
        if self.status.use_python_mode and not self.flags.force_pyobject:
            if len(self.lifted) > 0:
                warn_msg = 'Function "%s" was compiled in object mode without forceobj=True, but has lifted loops.' % self.func_attr.name,
            else:
                warn_msg = 'Function "%s" was compiled in object mode without forceobj=True.' % self.func_attr.name,
            warnings.warn_explicit(warn_msg, config.NumbaWarning,
                                   self.func_attr.filename,
                                   self.func_attr.lineno)

        return cr

    def _compile_bytecode(self):
        cres = self.frontend()
        if cres is not None:
            return cres
        return self.backend()


def compile_extra(typingctx, targetctx, func, args, return_type, flags,
                  locals):
    """
    Args
    ----
    - return_type
        Use ``None`` to indicate
    """
    pipeline = Pipeline(typingctx, targetctx, args, return_type, flags, locals)
    return pipeline.compile_extra(func)


def compile_bytecode(typingctx, targetctx, bc, args, return_type, flags,
                     locals, lifted=(),
                     func_attr=DEFAULT_FUNCTION_ATTRIBUTES):

    pipeline = Pipeline(typingctx, targetctx, args, return_type, flags, locals)
    return pipeline.compile_bytecode(bc=bc, lifted=lifted, func_attr=func_attr)


def _is_nopython_types(t):
    return t != types.pyobject and not isinstance(t, types.Dummy)


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
    """
    assert assume.return_argument_array_only

    if not isinstance(return_type, types.Array):
        return

    # Walk IR to discover all return statements
    retstmts = []
    for bid, blk in interp.blocks.items():
        for inst in blk.body:
            if isinstance(inst, ir.Return):
                retstmts.append(inst)

    assert retstmts, "No return statemants?"

    # FIXME: In the future, we can return an array that is either a dynamically
    #        allocated array or an array that is passed as argument.  This
    #        must be statically resolvable.

    # The return value must be the first modification of the value.
    arguments = frozenset("%s.1" % arg for arg in interp.argspec.args)

    for ret in retstmts:
        if ret.value.name not in arguments:
            raise TypeError("Only accept returning of array passed into the "
                            "function as argument")

    # Legalized; tag return handling
    targetctx.metadata['return.array'] = 'arg'


def translate_stage(bytecode):
    interp = interpreter.Interpreter(bytecode=bytecode)

    if config.DUMP_CFG:
        interp.cfa.dump()

    interp.interpret()

    if config.DEBUG:
        interp.dump()

    macro.expand_macros(interp.blocks)

    if config.DUMP_IR:
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


def native_lowering_stage(targetctx, interp, typemap, restype, calltypes,
                          nocompile):
    # Lowering
    fndesc = lowering.PythonFunctionDescriptor.from_specialized_function(
        interp, typemap, restype, calltypes, mangler=targetctx.mangler)

    lower = lowering.Lower(targetctx, fndesc, interp)
    lower.lower()

    if nocompile:
        return None, lower.module, lower.function, fndesc
    else:
        # Prepare for execution
        cfunc = targetctx.get_executable(lower.function, fndesc, lower.env)

        targetctx.insert_user_function(cfunc, fndesc)

        return cfunc, lower.module, lower.function, fndesc


def py_lowering_stage(targetctx, interp, nocompile):
    fndesc = lowering.PythonFunctionDescriptor.from_object_mode_function(interp)
    lower = objmode.PyLower(targetctx, fndesc, interp)
    lower.lower()

    if nocompile:
        return None, lower.module, lower.function, fndesc
    else:
        cfunc = targetctx.get_executable(lower.function, fndesc, lower.env)
        return cfunc, lower.module, lower.function, fndesc


def ir_optimize_for_py_stage(interp):
    """
    This passes breaks semantic for the type inferer but they reduces
    refct calls for object mode.
    """
    irpasses.RemoveRedundantAssign(interp).run()
    if config.DEBUG:
        print("ir optimize".center(80, '-'))
        interp.dump()
