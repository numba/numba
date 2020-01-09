from __future__ import print_function, division, absolute_import

from collections import namedtuple
import sys
import copy
import warnings
from .tracing import event

from numba import (bytecode, interpreter, postproc, typing,  utils, config,
                   errors,)
from numba.targets import cpu, callconv
from numba.parfor import ParforDiagnostics
from numba.inline_closurecall import InlineClosureCallPass
from numba.errors import CompilerError

from .compiler_machinery import PassManager

from .untyped_passes import (ExtractByteCode, TranslateByteCode, FixupArgs,
                             IRProcessing, DeadBranchPrune,
                             RewriteSemanticConstants, InlineClosureLikes,
                             GenericRewrites, WithLifting, InlineInlinables,
                             FindLiterallyCalls, MakeFunctionToJitFunction,
                             CanonicalizeLoopExit, CanonicalizeLoopEntry,
                             LiteralUnroll,
                             )

from .typed_passes import (NopythonTypeInference, AnnotateTypes,
                           NopythonRewrites, PreParforPass, ParforPass,
                           DumpParforDiagnostics, IRLegalization,
                           NoPythonBackend, InlineOverloads)

from .object_mode_passes import (ObjectModeFrontEnd, ObjectModeBackEnd,
                                 CompileInterpMode)


class Flags(utils.ConfigOptions):
    # These options are all false by default, but the defaults are
    # different with the @jit decorator (see targets.options.TargetOptions).

    OPTIONS = {
        # Enable loop-lifting
        'enable_looplift': False,
        # Enable pyobject mode (in general)
        'enable_pyobject': False,
        # Enable pyobject mode inside lifted loops
        'enable_pyobject_looplift': False,
        # Force pyobject mode inside the whole function
        'force_pyobject': False,
        # Release GIL inside the native function
        'release_gil': False,
        'no_compile': False,
        'debuginfo': False,
        'boundscheck': False,
        'forceinline': False,
        'no_cpython_wrapper': False,
        # Enable automatic parallel optimization, can be fine-tuned by taking
        # a dictionary of sub-options instead of a boolean, see parfor.py for
        # detail.
        'auto_parallel': cpu.ParallelOptions(False),
        'nrt': False,
        'no_rewrites': False,
        'error_model': 'python',
        'fastmath': cpu.FastMathOptions(False),
        'noalias': False,
        'inline': cpu.InlineOptions('never'),
    }


DEFAULT_FLAGS = Flags()
DEFAULT_FLAGS.set('nrt')


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
             "call_helper",
             "environment",
             "metadata",
             # List of functions to call to initialize on unserialization
             # (i.e cache load).
             "reload_init",
             ]


class CompileResult(namedtuple("_CompileResult", CR_FIELDS)):
    """
    A structure holding results from the compilation of a function.
    """

    __slots__ = ()

    def _reduce(self):
        """
        Reduce a CompileResult to picklable components.
        """
        libdata = self.library.serialize_using_object_code()
        # Make it (un)picklable efficiently
        typeann = str(self.type_annotation)
        fndesc = self.fndesc
        # Those don't need to be pickled and may fail
        fndesc.typemap = fndesc.calltypes = None

        return (libdata, self.fndesc, self.environment, self.signature,
                self.objectmode, self.interpmode, self.lifted, typeann,
                self.reload_init)

    @classmethod
    def _rebuild(cls, target_context, libdata, fndesc, env,
                 signature, objectmode, interpmode, lifted, typeann,
                 reload_init):
        if reload_init:
            # Re-run all
            for fn in reload_init:
                fn()

        library = target_context.codegen().unserialize_library(libdata)
        cfunc = target_context.get_executable(library, fndesc, env)
        cr = cls(target_context=target_context,
                 typing_context=target_context.typing_context,
                 library=library,
                 environment=env,
                 entry_point=cfunc,
                 fndesc=fndesc,
                 type_annotation=typeann,
                 signature=signature,
                 objectmode=objectmode,
                 interpmode=interpmode,
                 lifted=lifted,
                 typing_error=None,
                 call_helper=None,
                 metadata=None,  # Do not store, arbitrary & potentially large!
                 reload_init=reload_init,
                 )
        return cr


_LowerResult = namedtuple("_LowerResult", [
    "fndesc",
    "call_helper",
    "cfunc",
    "env",
])


def compile_result(**kws):
    keys = set(kws.keys())
    fieldset = set(CR_FIELDS)
    badnames = keys - fieldset
    if badnames:
        raise NameError(*badnames)
    missing = fieldset - keys
    for k in missing:
        kws[k] = None
    # Avoid keeping alive traceback variables
    if sys.version_info >= (3,):
        err = kws['typing_error']
        if err is not None:
            kws['typing_error'] = err.with_traceback(None)
    return CompileResult(**kws)


def compile_isolated(func, args, return_type=None, flags=DEFAULT_FLAGS,
                     locals={}):
    """
    Compile the function in an isolated environment (typing and target
    context).
    Good for testing.
    """
    from .targets.registry import cpu_target
    typingctx = typing.Context()
    targetctx = cpu.CPUContext(typingctx)
    # Register the contexts in case for nested @jit or @overload calls
    with cpu_target.nested_context(typingctx, targetctx):
        return compile_extra(typingctx, targetctx, func, args, return_type,
                             flags, locals)


def run_frontend(func, inline_closures=False):
    """
    Run the compiler frontend over the given Python function, and return
    the function's canonical Numba IR.

    If inline_closures is Truthy then closure inlining will be run
    """
    # XXX make this a dedicated Pipeline?
    func_id = bytecode.FunctionIdentity.from_function(func)
    interp = interpreter.Interpreter(func_id)
    bc = bytecode.ByteCode(func_id=func_id)
    func_ir = interp.interpret(bc)
    if inline_closures:
        inline_pass = InlineClosureCallPass(func_ir, cpu.ParallelOptions(False),
                                            {}, False)
        inline_pass.run()
    post_proc = postproc.PostProcessor(func_ir)
    post_proc.run()
    return func_ir


class _CompileStatus(object):
    """
    Describes the state of compilation. Used like a C record.
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
    """
    Raised to indicate that a pipeline has completed early
    """

    def __init__(self, result):
        self.result = result


class StateDict(dict):
    """
    A dictionary that has an overloaded getattr and setattr to permit getting
    and setting key/values through the use of attributes.
    """

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError(attr)

    def __setattr__(self, attr, value):
        self[attr] = value


def _make_subtarget(targetctx, flags):
    """
    Make a new target context from the given target context and flags.
    """
    subtargetoptions = {}
    if flags.debuginfo:
        subtargetoptions['enable_debuginfo'] = True
    if flags.boundscheck:
        subtargetoptions['enable_boundscheck'] = True
    if flags.nrt:
        subtargetoptions['enable_nrt'] = True
    if flags.auto_parallel:
        subtargetoptions['auto_parallel'] = flags.auto_parallel
    if flags.fastmath:
        subtargetoptions['fastmath'] = flags.fastmath
    error_model = callconv.create_error_model(flags.error_model, targetctx)
    subtargetoptions['error_model'] = error_model

    return targetctx.subtarget(**subtargetoptions)


class CompilerBase(object):
    """
    Stores and manages states for the compiler
    """

    def __init__(self, typingctx, targetctx, library, args, return_type, flags,
                 locals):
        # Make sure the environment is reloaded
        config.reload_config()
        typingctx.refresh()
        targetctx.refresh()

        self.state = StateDict()

        self.state.typingctx = typingctx
        self.state.targetctx = _make_subtarget(targetctx, flags)
        self.state.library = library
        self.state.args = args
        self.state.return_type = return_type
        self.state.flags = flags
        self.state.locals = locals

        # Results of various steps of the compilation pipeline
        self.state.bc = None
        self.state.func_id = None
        self.state.func_ir = None
        self.state.lifted = None
        self.state.lifted_from = None
        self.state.typemap = None
        self.state.calltypes = None
        self.state.type_annotation = None
        # holds arbitrary inter-pipeline stage meta data
        self.state.metadata = {}
        self.state.reload_init = []
        # hold this for e.g. with_lifting, null out on exit
        self.state.pipeline = self

        # parfor diagnostics info, add to metadata
        self.state.parfor_diagnostics = ParforDiagnostics()
        self.state.metadata['parfor_diagnostics'] = \
            self.state.parfor_diagnostics

        self.state.status = _CompileStatus(
            can_fallback=self.state.flags.enable_pyobject,
            can_giveup=config.COMPATIBILITY_MODE
        )

    def compile_extra(self, func):
        self.state.func_id = bytecode.FunctionIdentity.from_function(func)
        try:
            ExtractByteCode().run_pass(self.state)
        except Exception as e:
            if self.state.status.can_giveup:
                CompileInterpMode().run_pass(self.state)
                return self.state.cr
            else:
                raise e

        self.state.lifted = ()
        self.state.lifted_from = None
        return self._compile_bytecode()

    def compile_ir(self, func_ir, lifted=(), lifted_from=None):
        self.state.func_id = func_ir.func_id
        self.state.lifted = lifted
        self.state.lifted_from = lifted_from
        self.state.func_ir = func_ir
        self.state.nargs = self.state.func_ir.arg_count

        FixupArgs().run_pass(self.state)
        return self._compile_ir()

    def define_pipelines(self):
        """Child classes override this to customize the pipelines in use.
        """
        raise NotImplementedError()

    def _compile_core(self):
        """
        Populate and run compiler pipeline
        """
        pms = self.define_pipelines()
        for pm in pms:
            pipeline_name = pm.pipeline_name
            func_name = "%s.%s" % (self.state.func_id.modname,
                                   self.state.func_id.func_qualname)

            event("Pipeline: %s for %s" % (pipeline_name, func_name))
            self.state.metadata['pipeline_times'] = {pipeline_name:
                                                     pm.exec_times}
            is_final_pipeline = pm == pms[-1]
            res = None
            try:
                pm.run(self.state)
                if self.state.cr is not None:
                    break
            except _EarlyPipelineCompletion as e:
                res = e.result
                break
            except Exception as e:
                self.state.status.fail_reason = e
                if is_final_pipeline:
                    raise e
        else:
            raise CompilerError("All available pipelines exhausted")

        # Pipeline is done, remove self reference to release refs to user code
        self.state.pipeline = None

        # organise a return
        if res is not None:
            # Early pipeline completion
            return res
        else:
            assert self.state.cr is not None
            return self.state.cr

    def _compile_bytecode(self):
        """
        Populate and run pipeline for bytecode input
        """
        assert self.state.func_ir is None
        return self._compile_core()

    def _compile_ir(self):
        """
        Populate and run pipeline for IR input
        """
        assert self.state.func_ir is not None
        return self._compile_core()


class Compiler(CompilerBase):
    """The default compiler
    """

    def define_pipelines(self):
        # this maintains the objmode fallback behaviour
        pms = []
        if not self.state.flags.force_pyobject:
            pms.append(DefaultPassBuilder.define_nopython_pipeline(self.state))
        if self.state.status.can_fallback or self.state.flags.force_pyobject:
            pms.append(
                DefaultPassBuilder.define_objectmode_pipeline(self.state)
            )
        if self.state.status.can_giveup:
            pms.append(
                DefaultPassBuilder.define_interpreted_pipeline(self.state)
            )
        return pms


class DefaultPassBuilder(object):
    """
    This is the default pass builder, it contains the "classic" default
    pipelines as pre-canned PassManager instances:
      - nopython
      - objectmode
      - interpreted
    """
    @staticmethod
    def define_nopython_pipeline(state, name='nopython'):
        """Returns an nopython mode pipeline based PassManager
        """
        pm = PassManager(name)
        if state.func_ir is None:
            pm.add_pass(TranslateByteCode, "analyzing bytecode")
            pm.add_pass(FixupArgs, "fix up args")
        pm.add_pass(IRProcessing, "processing IR")
        pm.add_pass(WithLifting, "Handle with contexts")

        # pre typing
        if not state.flags.no_rewrites:
            pm.add_pass(RewriteSemanticConstants, "rewrite semantic constants")
            pm.add_pass(DeadBranchPrune, "dead branch pruning")
            pm.add_pass(GenericRewrites, "nopython rewrites")

        pm.add_pass(InlineClosureLikes,
                    "inline calls to locally defined closures")
        # convert any remaining closures into functions
        pm.add_pass(MakeFunctionToJitFunction,
                    "convert make_function into JIT functions")
        # inline functions that have been determined as inlinable and rerun
        # branch pruning, this needs to be run after closures are inlined as
        # the IR repr of a closure masks call sites if an inlinable is called
        # inside a closure
        pm.add_pass(InlineInlinables, "inline inlinable functions")
        if not state.flags.no_rewrites:
            pm.add_pass(DeadBranchPrune, "dead branch pruning")

        pm.add_pass(FindLiterallyCalls, "find literally calls")
        pm.add_pass(LiteralUnroll, "handles literal_unroll")

        # typing
        pm.add_pass(NopythonTypeInference, "nopython frontend")
        pm.add_pass(AnnotateTypes, "annotate types")

        # optimisation
        pm.add_pass(InlineOverloads, "inline overloaded functions")
        if state.flags.auto_parallel.enabled:
            pm.add_pass(PreParforPass, "Preprocessing for parfors")
        if not state.flags.no_rewrites:
            pm.add_pass(NopythonRewrites, "nopython rewrites")
        if state.flags.auto_parallel.enabled:
            pm.add_pass(ParforPass, "convert to parfors")

        # legalise
        pm.add_pass(IRLegalization,
                    "ensure IR is legal prior to lowering")

        # lower
        pm.add_pass(NoPythonBackend, "nopython mode backend")
        pm.add_pass(DumpParforDiagnostics, "dump parfor diagnostics")
        pm.finalize()
        return pm

    @staticmethod
    def define_objectmode_pipeline(state, name='object'):
        """Returns an object-mode pipeline based PassManager
        """
        pm = PassManager(name)
        if state.func_ir is None:
            pm.add_pass(TranslateByteCode, "analyzing bytecode")
            pm.add_pass(FixupArgs, "fix up args")
        pm.add_pass(IRProcessing, "processing IR")

        if utils.PYVERSION >= (3, 7):
            # The following passes are needed to adjust for looplifting
            pm.add_pass(CanonicalizeLoopEntry, "canonicalize loop entry")
            pm.add_pass(CanonicalizeLoopExit, "canonicalize loop exit")

        pm.add_pass(ObjectModeFrontEnd, "object mode frontend")
        pm.add_pass(InlineClosureLikes,
                    "inline calls to locally defined closures")
        # convert any remaining closures into functions
        pm.add_pass(MakeFunctionToJitFunction,
                    "convert make_function into JIT functions")
        pm.add_pass(AnnotateTypes, "annotate types")
        pm.add_pass(IRLegalization, "ensure IR is legal prior to lowering")
        pm.add_pass(ObjectModeBackEnd, "object mode backend")
        pm.finalize()
        return pm

    @staticmethod
    def define_interpreted_pipeline(state, name="interpreted"):
        """Returns an interpreted mode pipeline based PassManager
        """
        pm = PassManager(name)
        pm.add_pass(CompileInterpMode,
                    "compiling with interpreter mode")
        pm.finalize()
        return pm


def compile_extra(typingctx, targetctx, func, args, return_type, flags,
                  locals, library=None, pipeline_class=Compiler):
    """Compiler entry point

    Parameter
    ---------
    typingctx :
        typing context
    targetctx :
        target context
    func : function
        the python function to be compiled
    args : tuple, list
        argument types
    return_type :
        Use ``None`` to indicate void return
    flags : numba.compiler.Flags
        compiler flags
    library : numba.codegen.CodeLibrary
        Used to store the compiled code.
        If it is ``None``, a new CodeLibrary is used.
    pipeline_class : type like numba.compiler.CompilerBase
        compiler pipeline
    """
    pipeline = pipeline_class(typingctx, targetctx, library,
                              args, return_type, flags, locals)
    return pipeline.compile_extra(func)


def compile_ir(typingctx, targetctx, func_ir, args, return_type, flags,
               locals, lifted=(), lifted_from=None, is_lifted_loop=False,
               library=None, pipeline_class=Compiler):
    """
    Compile a function with the given IR.

    For internal use only.
    """

    # This is a special branch that should only run on IR from a lifted loop
    if is_lifted_loop:
        # This code is pessimistic and costly, but it is a not often trodden
        # path and it will go away once IR is made immutable. The problem is
        # that the rewrite passes can mutate the IR into a state that makes
        # it possible for invalid tokens to be transmitted to lowering which
        # then trickle through into LLVM IR and causes RuntimeErrors as LLVM
        # cannot compile it. As a result the following approach is taken:
        # 1. Create some new flags that copy the original ones but switch
        #    off rewrites.
        # 2. Compile with 1. to get a compile result
        # 3. Try and compile another compile result but this time with the
        #    original flags (and IR being rewritten).
        # 4. If 3 was successful, use the result, else use 2.

        # create flags with no rewrites
        norw_flags = copy.deepcopy(flags)
        norw_flags.no_rewrites = True

        def compile_local(the_ir, the_flags):
            pipeline = pipeline_class(typingctx, targetctx, library,
                                      args, return_type, the_flags, locals)
            return pipeline.compile_ir(func_ir=the_ir, lifted=lifted,
                                       lifted_from=lifted_from)

        # compile with rewrites off, IR shouldn't be mutated irreparably
        norw_cres = compile_local(func_ir.copy(), norw_flags)

        # try and compile with rewrites on if no_rewrites was not set in the
        # original flags, IR might get broken but we've got a CompileResult
        # that's usable from above.
        rw_cres = None
        if not flags.no_rewrites:
            # Suppress warnings in compilation retry
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", errors.NumbaWarning)
                try:
                    rw_cres = compile_local(func_ir.copy(), flags)
                except Exception:
                    pass
        # if the rewrite variant of compilation worked, use it, else use
        # the norewrites backup
        if rw_cres is not None:
            cres = rw_cres
        else:
            cres = norw_cres
        return cres

    else:
        pipeline = pipeline_class(typingctx, targetctx, library,
                                  args, return_type, flags, locals)
        return pipeline.compile_ir(func_ir=func_ir, lifted=lifted,
                                   lifted_from=lifted_from)


def compile_internal(typingctx, targetctx, library,
                     func, args, return_type, flags, locals):
    """
    For internal use only.
    """
    pipeline = Compiler(typingctx, targetctx, library,
                        args, return_type, flags, locals)
    return pipeline.compile_extra(func)
