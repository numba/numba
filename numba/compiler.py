from __future__ import print_function, division, absolute_import

from contextlib import contextmanager
from collections import namedtuple, defaultdict
import sys
import warnings
import traceback
from .tracing import event

from numba import (bytecode, interpreter, funcdesc, postproc,
                   typing, typeinfer, lowering, pylowering, utils, config,
                   errors, types, ir, rewrites, transforms)
from numba.targets import cpu, callconv
from numba.annotations import type_annotations
from numba.parfor import PreParforPass, ParforPass, Parfor, ParforDiagnostics
from numba.inline_closurecall import InlineClosureCallPass
from numba.errors import CompilerError
from numba.ir_utils import raise_on_unsupported_feature
from numba.compiler_lock import global_compiler_lock
from numba.analysis import dead_branch_prune

# terminal color markup
_termcolor = errors.termcolor()


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
        'boundcheck': False,
        'forceinline': False,
        'no_cpython_wrapper': False,
        # Enable automatic parallel optimization, can be fine-tuned by taking
        # a dictionary of sub-options instead of a boolean, see parfor.py for
        # detail.
        'auto_parallel': cpu.ParallelOptions(False),
        'nrt': False,
        'no_rewrites': False,
        'error_model': 'python',
        'fastmath': False,
        'noalias': False,
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
             "metadata",]


class CompileResult(namedtuple("_CompileResult", CR_FIELDS)):
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
                self.objectmode, self.interpmode, self.lifted, typeann)

    @classmethod
    def _rebuild(cls, target_context, libdata, fndesc, env,
                 signature, objectmode, interpmode, lifted, typeann):
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
                 metadata=None, # Do not store, arbitrary and potentially large!
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


def run_frontend(func):
    """
    Run the compiler frontend over the given Python function, and return
    the function's canonical Numba IR.
    """
    # XXX make this a dedicated Pipeline?
    func_id = bytecode.FunctionIdentity.from_function(func)
    interp = interpreter.Interpreter(func_id)
    bc = bytecode.ByteCode(func_id=func_id)
    func_ir = interp.interpret(bc)
    post_proc = postproc.PostProcessor(func_ir)
    post_proc.run()
    return func_ir


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

        # For python2, attach the traceback of the previous exception.
        if not utils.IS_PY3 and config.FULL_TRACEBACKS:
            # strip the new message to just print the error string and not
            # the marked up source etc (this is handled already).
            stripped = _termcolor.errmsg(newmsg.split('\n')[1])
            fmt = "Caused By:\n{tb}\n{newmsg}"
            newmsg = fmt.format(tb=traceback.format_exc(), newmsg=stripped)

        exc.args = (newmsg,)
        return exc

    @global_compiler_lock
    def run(self, status):
        assert self._finalized, "PM must be finalized before run()"
        for pipeline_name in self.pipeline_order:
            event(pipeline_name)
            is_final_pipeline = pipeline_name == self.pipeline_order[-1]
            for stage, stage_name in self.pipeline_stages[pipeline_name]:
                try:
                    event(stage_name)
                    stage()
                except _EarlyPipelineCompletion as e:
                    return e.result
                except BaseException as e:
                    msg = "Failed in %s mode pipeline (step: %s)" % \
                        (pipeline_name, stage_name)
                    patched_exception = self._patch_error(msg, e)
                    # No more fallback pipelines?
                    if is_final_pipeline:
                        raise patched_exception
                    # Go to next fallback pipeline
                    else:
                        status.fail_reason = patched_exception
                        break
            else:
                return None

        # TODO save all error information
        raise CompilerError("All pipelines have failed")


class BasePipeline(object):
    """
    Stores and manages states for the compiler pipeline
    """
    def __init__(self, typingctx, targetctx, library, args, return_type, flags,
                 locals):
        # Make sure the environment is reloaded
        config.reload_config()
        typingctx.refresh()
        targetctx.refresh()

        self.typingctx = typingctx
        self.targetctx = _make_subtarget(targetctx, flags)
        self.library = library
        self.args = args
        self.return_type = return_type
        self.flags = flags
        self.locals = locals

        # Results of various steps of the compilation pipeline
        self.bc = None
        self.func_id = None
        self.func_ir = None
        self.func_ir_original = None  # used for fallback
        self.lifted = None
        self.lifted_from = None
        self.typemap = None
        self.calltypes = None
        self.type_annotation = None
        self.metadata = {} # holds arbitrary inter-pipeline stage meta data

        # parfor diagnostics info, add to metadata
        self.parfor_diagnostics = ParforDiagnostics()
        self.metadata['parfor_diagnostics'] = self.parfor_diagnostics

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
                                       errors.NumbaWarning,
                                       self.func_id.filename,
                                       self.func_id.firstlineno)

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
                                       errors.NumbaWarning,
                                       self.func_id.filename,
                                       self.func_id.firstlineno)

                raise

    def extract_bytecode(self, func_id):
        """
        Extract bytecode from function
        """
        bc = bytecode.ByteCode(func_id)
        if config.DUMP_BYTECODE:
            print(bc.dump())

        return bc

    def compile_extra(self, func):
        self.func_id = bytecode.FunctionIdentity.from_function(func)

        try:
            bc = self.extract_bytecode(self.func_id)
        except BaseException as e:
            if self.status.can_giveup:
                self.stage_compile_interp_mode()
                return self.cr
            else:
                raise e

        self.bc = bc
        self.lifted = ()
        self.lifted_from = None
        return self._compile_bytecode()

    def compile_ir(self, func_ir, lifted=(), lifted_from=None):
        self.func_id = func_ir.func_id
        self.lifted = lifted
        self.lifted_from = lifted_from

        self._set_and_check_ir(func_ir)
        return self._compile_ir()

    def stage_analyze_bytecode(self):
        """
        Analyze bytecode and translating to Numba IR
        """
        func_ir = translate_stage(self.func_id, self.bc)
        self._set_and_check_ir(func_ir)

    def _set_and_check_ir(self, func_ir):
        self.func_ir = func_ir
        self.nargs = self.func_ir.arg_count
        if not self.args and self.flags.force_pyobject:
            # Allow an empty argument types specification when object mode
            # is explicitly requested.
            self.args = (types.pyobject,) * self.nargs
        elif len(self.args) != self.nargs:
            raise TypeError("Signature mismatch: %d argument types given, "
                            "but function takes %d arguments"
                            % (len(self.args), self.nargs))

    def stage_process_ir(self):
        ir_processing_stage(self.func_ir)

    def stage_preserve_ir(self):
        self.func_ir_original = self.func_ir.copy()

    def frontend_looplift(self):
        """
        Loop lifting analysis and transformation
        """
        loop_flags = self.flags.copy()
        outer_flags = self.flags.copy()
        # Do not recursively loop lift
        outer_flags.unset('enable_looplift')
        loop_flags.unset('enable_looplift')
        if not self.flags.enable_pyobject_looplift:
            loop_flags.unset('enable_pyobject')

        main, loops = transforms.loop_lifting(self.func_ir,
                                              typingctx=self.typingctx,
                                              targetctx=self.targetctx,
                                              locals=self.locals,
                                              flags=loop_flags)
        if loops:
            # Some loops were extracted
            if config.DEBUG_FRONTEND or config.DEBUG:
                for loop in loops:
                    print("Lifting loop", loop.get_source_location())

            cres = compile_ir(self.typingctx, self.targetctx, main,
                              self.args, self.return_type,
                              outer_flags, self.locals,
                              lifted=tuple(loops), lifted_from=None)
            return cres

    def stage_frontend_withlift(self):
        """
        Extract with-contexts
        """
        main, withs = transforms.with_lifting(
            func_ir=self.func_ir,
            typingctx=self.typingctx,
            targetctx=self.targetctx,
            flags=self.flags,
            locals=self.locals,
            )
        if withs:
            cres = compile_ir(self.typingctx, self.targetctx, main,
                              self.args, self.return_type,
                              self.flags, self.locals,
                              lifted=tuple(withs), lifted_from=None,
                              pipeline_class=type(self))
            raise _EarlyPipelineCompletion(cres)

    def stage_objectmode_frontend(self):
        """
        Front-end: Analyze bytecode, generate Numba IR, infer types
        """
        self.func_ir = self.func_ir_original or self.func_ir
        if self.flags.enable_looplift:
            assert not self.lifted
            cres = self.frontend_looplift()
            if cres is not None:
                raise _EarlyPipelineCompletion(cres)

        # Fallback typing: everything is a python object
        self.typemap = defaultdict(lambda: types.pyobject)
        self.calltypes = defaultdict(lambda: types.pyobject)
        self.return_type = types.pyobject

    def stage_dead_branch_prune(self):
        """
        This prunes dead branches, a dead branch is one which is derivable as
        not taken at compile time purely based on const/literal evaluation.
        """
        assert self.func_ir
        msg = ('Internal error in pre-inference dead branch pruning '
               'pass encountered during compilation of '
               'function "%s"' % (self.func_id.func_name,))
        with self.fallback_context(msg):
            dead_branch_prune(self.func_ir, self.args)

        if config.DEBUG or config.DUMP_IR:
            print('branch_pruned_ir'.center(80, '-'))
            print(self.func_ir.dump())
            print('end branch_pruned_ir'.center(80, '-'))

    def stage_nopython_frontend(self):
        """
        Type inference and legalization
        """
        with self.fallback_context('Function "%s" failed type inference'
                                   % (self.func_id.func_name,)):
            # Type inference
            typemap, return_type, calltypes = type_inference_stage(
                self.typingctx,
                self.func_ir,
                self.args,
                self.return_type,
                self.locals)
            self.typemap = typemap
            self.return_type = return_type
            self.calltypes = calltypes

        with self.fallback_context('Function "%s" has invalid return type'
                                   % (self.func_id.func_name,)):
            legalize_return_type(self.return_type, self.func_ir,
                                 self.targetctx)

    def stage_generic_rewrites(self):
        """
        Perform any intermediate representation rewrites before type
        inference.
        """
        assert self.func_ir
        msg = ('Internal error in pre-inference rewriting '
               'pass encountered during compilation of '
               'function "%s"' % (self.func_id.func_name,))
        with self.fallback_context(msg):
            rewrites.rewrite_registry.apply('before-inference',
                                            self, self.func_ir)

    def stage_nopython_rewrites(self):
        """
        Perform any intermediate representation rewrites after type
        inference.
        """
        # Ensure we have an IR and type information.
        assert self.func_ir
        assert isinstance(getattr(self, 'typemap', None), dict)
        assert isinstance(getattr(self, 'calltypes', None), dict)
        msg = ('Internal error in post-inference rewriting '
               'pass encountered during compilation of '
               'function "%s"' % (self.func_id.func_name,))
        with self.fallback_context(msg):
            rewrites.rewrite_registry.apply('after-inference',
                                            self, self.func_ir)

    def stage_pre_parfor_pass(self):
        """
        Preprocessing for data-parallel computations.
        """
        # Ensure we have an IR and type information.
        assert self.func_ir
        preparfor_pass = PreParforPass(
            self.func_ir,
            self.type_annotation.typemap,
            self.type_annotation.calltypes, self.typingctx,
            self.flags.auto_parallel,
            self.parfor_diagnostics.replaced_fns
            )

        preparfor_pass.run()

    def stage_parfor_pass(self):
        """
        Convert data-parallel computations into Parfor nodes
        """
        # Ensure we have an IR and type information.
        assert self.func_ir
        parfor_pass = ParforPass(self.func_ir, self.type_annotation.typemap,
            self.type_annotation.calltypes, self.return_type, self.typingctx,
            self.flags.auto_parallel, self.flags, self.parfor_diagnostics)
        parfor_pass.run()

        if config.WARNINGS:
            # check the parfor pass worked and warn if it didn't
            has_parfor = False
            for blk in self.func_ir.blocks.values():
                for stmnt in blk.body:
                    if isinstance(stmnt, Parfor):
                        has_parfor = True
                        break
                else:
                    continue
                break

            if not has_parfor:
                # parfor calls the compiler chain again with a string
                if not self.func_ir.loc.filename == '<string>':
                    msg = ("parallel=True was specified but no transformation"
                           " for parallel execution was possible.")
                    warnings.warn_explicit(
                        msg,
                        errors.NumbaWarning,
                        self.func_id.filename,
                        self.func_id.firstlineno
                        )

    def stage_inline_pass(self):
        """
        Inline calls to locally defined closures.
        """
        # Ensure we have an IR and type information.
        assert self.func_ir
        inline_pass = InlineClosureCallPass(self.func_ir,
                                            self.flags.auto_parallel,
                                            self.parfor_diagnostics.replaced_fns)
        inline_pass.run()
        # Remove all Dels, and re-run postproc
        post_proc = postproc.PostProcessor(self.func_ir)
        post_proc.run()

        if config.DEBUG or config.DUMP_IR:
            name = self.func_ir.func_id.func_qualname
            print(("IR DUMP: %s" % name).center(80, "-"))
            self.func_ir.dump()

    def stage_annotate_type(self):
        """
        Create type annotation after type inference
        """
        self.type_annotation = type_annotations.TypeAnnotation(
            func_ir=self.func_ir,
            typemap=self.typemap,
            calltypes=self.calltypes,
            lifted=self.lifted,
            lifted_from=self.lifted_from,
            args=self.args,
            return_type=self.return_type,
            html_output=config.HTML)

        if config.ANNOTATE:
            print("ANNOTATION".center(80, '-'))
            print(self.type_annotation)
            print('=' * 80)
        if config.HTML:
            with open(config.HTML, 'w') as fout:
                self.type_annotation.html_annotate(fout)

    def stage_dump_diagnostics(self):
        if self.flags.auto_parallel.enabled:
            if config.PARALLEL_DIAGNOSTICS:
                if self.parfor_diagnostics is not None:
                    self.parfor_diagnostics.dump(config.PARALLEL_DIAGNOSTICS)
                else:
                    raise RuntimeError("Diagnostics failed.")

    def backend_object_mode(self):
        """
        Object mode compilation
        """
        with self.giveup_context("Function %s failed at object mode lowering"
                                 % (self.func_id.func_name,)):
            if len(self.args) != self.nargs:
                # append missing
                self.args = (tuple(self.args) + (types.pyobject,) *
                             (self.nargs - len(self.args)))

            return py_lowering_stage(self.targetctx,
                                     self.library,
                                     self.func_ir,
                                     self.flags)

    def backend_nopython_mode(self):
        """Native mode compilation"""
        msg = ("Function %s failed at nopython "
               "mode lowering" % (self.func_id.func_name,))
        with self.fallback_context(msg):
            return native_lowering_stage(
                self.targetctx,
                self.library,
                self.func_ir,
                self.typemap,
                self.return_type,
                self.calltypes,
                self.flags,
                self.metadata)

    def _backend(self, lowerfn, objectmode):
        """
        Back-end: Generate LLVM IR from Numba IR, compile to machine code
        """
        if self.library is None:
            codegen = self.targetctx.codegen()
            self.library = codegen.create_library(self.func_id.func_qualname)
            # Enable object caching upfront, so that the library can
            # be later serialized.
            self.library.enable_object_caching()

        lowered = lowerfn()
        signature = typing.signature(self.return_type, *self.args)
        self.cr = compile_result(
            typing_context=self.typingctx,
            target_context=self.targetctx,
            entry_point=lowered.cfunc,
            typing_error=self.status.fail_reason,
            type_annotation=self.type_annotation,
            library=self.library,
            call_helper=lowered.call_helper,
            signature=signature,
            objectmode=objectmode,
            interpmode=False,
            lifted=self.lifted,
            fndesc=lowered.fndesc,
            environment=lowered.env,
            metadata=self.metadata,
            )

    def stage_objectmode_backend(self):
        """
        Lowering for object mode
        """
        lowerfn = self.backend_object_mode
        self._backend(lowerfn, objectmode=True)

        # Warn if compiled function in object mode and force_pyobject not set
        if not self.flags.force_pyobject:
            if len(self.lifted) > 0:
                warn_msg = ('Function "%s" was compiled in object mode without'
                            ' forceobj=True, but has lifted loops.' %
                            (self.func_id.func_name,))
            else:
                warn_msg = ('Function "%s" was compiled in object mode without'
                            ' forceobj=True.' % (self.func_id.func_name,))
            warnings.warn_explicit(warn_msg, errors.NumbaWarning,
                                   self.func_id.filename,
                                   self.func_id.firstlineno)
            if self.flags.release_gil:
                warn_msg = ("Code running in object mode won't allow parallel"
                            " execution despite nogil=True.")
                warnings.warn_explicit(warn_msg, errors.NumbaWarning,
                                       self.func_id.filename,
                                       self.func_id.firstlineno)

    def stage_nopython_backend(self):
        """
        Do lowering for nopython
        """
        lowerfn = self.backend_nopython_mode
        self._backend(lowerfn, objectmode=False)

    def stage_compile_interp_mode(self):
        """
        Just create a compile result for interpreter mode
        """
        args = [types.pyobject] * len(self.args)
        signature = typing.signature(types.pyobject, *args)
        self.cr = compile_result(typing_context=self.typingctx,
                                 target_context=self.targetctx,
                                 entry_point=self.func_id.func,
                                 typing_error=self.status.fail_reason,
                                 type_annotation="<Interpreter mode function>",
                                 signature=signature,
                                 objectmode=False,
                                 interpmode=True,
                                 lifted=(),
                                 fndesc=None,)

    def stage_ir_legalization(self):
        raise_on_unsupported_feature(self.func_ir)

    def stage_cleanup(self):
        """
        Cleanup intermediate results to release resources.
        """

    def define_pipelines(self, pm):
        """Child classes override this to customize the pipeline.
        """
        raise NotImplementedError()

    def add_preprocessing_stage(self, pm):
        """Add the preprocessing stage that analyzes the bytecode to prepare
        the Numba IR.
        """
        if self.func_ir is None:
            pm.add_stage(self.stage_analyze_bytecode, "analyzing bytecode")
        pm.add_stage(self.stage_process_ir, "processing IR")

    def add_pre_typing_stage(self, pm):
        """Add any stages that go before type-inference.
        The current stages contain type-agnostic rewrite passes.
        """
        if not self.flags.no_rewrites:
            if self.status.can_fallback:
                pm.add_stage(self.stage_preserve_ir,
                             "preserve IR for fallback")
            pm.add_stage(self.stage_generic_rewrites, "nopython rewrites")
            pm.add_stage(self.stage_dead_branch_prune, "dead branch pruning")
        pm.add_stage(self.stage_inline_pass,
                     "inline calls to locally defined closures")

    def add_typing_stage(self, pm):
        """Add the type-inference stage necessary for nopython mode.
        """
        pm.add_stage(self.stage_nopython_frontend, "nopython frontend")
        pm.add_stage(self.stage_annotate_type, "annotate type")

    def add_optimization_stage(self, pm):
        """Add optimization stages.
        """
        if self.flags.auto_parallel.enabled:
            pm.add_stage(self.stage_pre_parfor_pass,
                         "Preprocessing for parfors")
        if not self.flags.no_rewrites:
            pm.add_stage(self.stage_nopython_rewrites, "nopython rewrites")
        if self.flags.auto_parallel.enabled:
            pm.add_stage(self.stage_parfor_pass, "convert to parfors")

    def add_lowering_stage(self, pm):
        """Add the lowering (code-generation) stage for nopython-mode
        """
        pm.add_stage(self.stage_nopython_backend, "nopython mode backend")

    def add_cleanup_stage(self, pm):
        """Add the clean-up stage to remove intermediate results.
        """
        pm.add_stage(self.stage_cleanup, "cleanup intermediate results")

    def add_with_handling_stage(self, pm):
        pm.add_stage(self.stage_frontend_withlift, "Handle with contexts")

    def define_nopython_pipeline(self, pm, name='nopython'):
        """Add the nopython-mode pipeline to the pipeline manager
        """
        pm.create_pipeline(name)
        self.add_preprocessing_stage(pm)
        self.add_with_handling_stage(pm)
        self.add_pre_typing_stage(pm)
        self.add_typing_stage(pm)
        self.add_optimization_stage(pm)
        pm.add_stage(self.stage_ir_legalization,
                     "ensure IR is legal prior to lowering")
        self.add_lowering_stage(pm)
        pm.add_stage(self.stage_dump_diagnostics, "dump diagnostics")
        self.add_cleanup_stage(pm)

    def define_objectmode_pipeline(self, pm, name='object'):
        """Add the object-mode pipeline to the pipeline manager
        """
        pm.create_pipeline(name)
        self.add_preprocessing_stage(pm)
        pm.add_stage(self.stage_objectmode_frontend,
                     "object mode frontend")
        pm.add_stage(self.stage_inline_pass,
                     "inline calls to locally defined closures")
        pm.add_stage(self.stage_annotate_type, "annotate type")
        pm.add_stage(self.stage_ir_legalization,
                     "ensure IR is legal prior to lowering")
        pm.add_stage(self.stage_objectmode_backend, "object mode backend")
        self.add_cleanup_stage(pm)

    def define_interpreted_pipeline(self, pm, name="interp"):
        """Add the interpreted-mode (fallback) pipeline to the pipeline manager
        """
        pm.create_pipeline(name)
        pm.add_stage(self.stage_compile_interp_mode,
                     "compiling with interpreter mode")
        self.add_cleanup_stage(pm)

    def _compile_core(self):
        """
        Populate and run compiler pipeline
        """
        pm = _PipelineManager()
        self.define_pipelines(pm)
        pm.finalize()
        res = pm.run(self.status)
        if res is not None:
            # Early pipeline completion
            return res
        else:
            assert self.cr is not None
            return self.cr

    def _compile_bytecode(self):
        """
        Populate and run pipeline for bytecode input
        """
        assert self.func_ir is None
        return self._compile_core()

    def _compile_ir(self):
        """
        Populate and run pipeline for IR input
        """
        assert self.func_ir is not None
        return self._compile_core()


class Pipeline(BasePipeline):
    """The default compiler pipeline
    """
    def define_pipelines(self, pm):
        if not self.flags.force_pyobject:
            self.define_nopython_pipeline(pm)
        if self.status.can_fallback or self.flags.force_pyobject:
            self.define_objectmode_pipeline(pm)
        if self.status.can_giveup:
            self.define_interpreted_pipeline(pm)


def _make_subtarget(targetctx, flags):
    """
    Make a new target context from the given target context and flags.
    """
    subtargetoptions = {}
    if flags.debuginfo:
        subtargetoptions['enable_debuginfo'] = True
    if flags.boundcheck:
        subtargetoptions['enable_boundcheck'] = True
    if flags.nrt:
        subtargetoptions['enable_nrt'] = True
    if flags.auto_parallel:
        subtargetoptions['auto_parallel'] = flags.auto_parallel
    if flags.fastmath:
        subtargetoptions['enable_fastmath'] = True
    error_model = callconv.create_error_model(flags.error_model, targetctx)
    subtargetoptions['error_model'] = error_model

    return targetctx.subtarget(**subtargetoptions)


def compile_extra(typingctx, targetctx, func, args, return_type, flags,
                  locals, library=None, pipeline_class=Pipeline):
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
    pipeline_class : type like numba.compiler.BasePipeline
        compiler pipeline
    """
    pipeline = pipeline_class(typingctx, targetctx, library,
                              args, return_type, flags, locals)
    return pipeline.compile_extra(func)


def compile_ir(typingctx, targetctx, func_ir, args, return_type, flags,
               locals, lifted=(), lifted_from=None, library=None,
               pipeline_class=Pipeline):
    """
    Compile a function with the given IR.

    For internal use only.
    """

    pipeline = pipeline_class(typingctx, targetctx, library,
                              args, return_type, flags, locals)
    return pipeline.compile_ir(func_ir=func_ir, lifted=lifted,
                               lifted_from=lifted_from)


def compile_internal(typingctx, targetctx, library,
                     func, args, return_type, flags, locals):
    """
    For internal use only.
    """
    pipeline = Pipeline(typingctx, targetctx, library,
                        args, return_type, flags, locals)
    return pipeline.compile_extra(func)


def legalize_return_type(return_type, interp, targetctx):
    """
    Only accept array return type iff it is passed into the function.
    Reject function object return types if in nopython mode.
    """
    if not targetctx.enable_nrt and isinstance(return_type, types.Array):
        # Walk IR to discover all arguments and all return statements
        retstmts = []
        caststmts = {}
        argvars = set()
        for bid, blk in interp.blocks.items():
            for inst in blk.body:
                if isinstance(inst, ir.Return):
                    retstmts.append(inst.value.name)
                elif isinstance(inst, ir.Assign):
                    if (isinstance(inst.value, ir.Expr)
                            and inst.value.op == 'cast'):
                        caststmts[inst.target.name] = inst.value
                    elif isinstance(inst.value, ir.Arg):
                        argvars.add(inst.target.name)

        assert retstmts, "No return statements?"

        for var in retstmts:
            cast = caststmts.get(var)
            if cast is None or cast.value.name not in argvars:
                raise TypeError("Only accept returning of array passed into "
                                "the function as argument")

    elif (isinstance(return_type, types.Function) or
            isinstance(return_type, types.Phantom)):
        msg = "Can't return function object ({}) in nopython mode"
        raise TypeError(msg.format(return_type))


def translate_stage(func_id, bytecode):
    interp = interpreter.Interpreter(func_id)
    return interp.interpret(bytecode)


def ir_processing_stage(func_ir):
    post_proc = postproc.PostProcessor(func_ir)
    post_proc.run()

    if config.DEBUG or config.DUMP_IR:
        name = func_ir.func_id.func_qualname
        print(("IR DUMP: %s" % name).center(80, "-"))
        func_ir.dump()
        if func_ir.is_generator:
            print(("GENERATOR INFO: %s" % name).center(80, "-"))
            func_ir.dump_generator_info()

    return func_ir


def type_inference_stage(typingctx, interp, args, return_type, locals={}):
    if len(args) != interp.arg_count:
        raise TypeError("Mismatch number of argument types")

    warnings = errors.WarningsFixer(errors.NumbaWarning)
    infer = typeinfer.TypeInferer(typingctx, interp, warnings)
    with typingctx.callstack.register(infer, interp.func_id, args):
        # Seed argument types
        for index, (name, ty) in enumerate(zip(interp.arg_names, args)):
            infer.seed_argument(name, index, ty)

        # Seed return type
        if return_type is not None:
            infer.seed_return(return_type)

        # Seed local types
        for k, v in locals.items():
            infer.seed_type(k, v)

        infer.build_constraint()
        infer.propagate()
        typemap, restype, calltypes = infer.unify()

    # Output all Numba warnings
    warnings.flush()

    return typemap, restype, calltypes


def native_lowering_stage(targetctx, library, interp, typemap, restype,
                          calltypes, flags, metadata):
    # Lowering
    fndesc = funcdesc.PythonFunctionDescriptor.from_specialized_function(
        interp, typemap, restype, calltypes, mangler=targetctx.mangler,
        inline=flags.forceinline, noalias=flags.noalias)

    with targetctx.push_code_library(library):
        lower = lowering.Lower(targetctx, library, fndesc, interp,
                               metadata=metadata)
        lower.lower()
        if not flags.no_cpython_wrapper:
            lower.create_cpython_wrapper(flags.release_gil)
        env = lower.env
        call_helper = lower.call_helper
        del lower

    if flags.no_compile:
        return _LowerResult(fndesc, call_helper, cfunc=None, env=env)
    else:
        # Prepare for execution
        cfunc = targetctx.get_executable(library, fndesc, env)
        # Insert native function for use by other jitted-functions.
        # We also register its library to allow for inlining.
        targetctx.insert_user_function(cfunc, fndesc, [library])
        return _LowerResult(fndesc, call_helper, cfunc=cfunc, env=env)


def py_lowering_stage(targetctx, library, interp, flags):
    fndesc = funcdesc.PythonFunctionDescriptor.from_object_mode_function(
        interp
        )
    with targetctx.push_code_library(library):
        lower = pylowering.PyLower(targetctx, library, fndesc, interp)
        lower.lower()
        if not flags.no_cpython_wrapper:
            lower.create_cpython_wrapper()
        env = lower.env
        call_helper = lower.call_helper
        del lower

    if flags.no_compile:
        return _LowerResult(fndesc, call_helper, cfunc=None, env=env)
    else:
        # Prepare for execution
        cfunc = targetctx.get_executable(library, fndesc, env)
        return _LowerResult(fndesc, call_helper, cfunc=cfunc, env=env)
