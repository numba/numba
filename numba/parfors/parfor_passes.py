
import warnings

from numba.core import (errors, config)

from numba.parfors.parfor import PreParforPass as _parfor_PreParforPass
from numba.parfors.parfor import ParforPass as _parfor_ParforPass
from numba.parfors.parfor import ParforFusionPass as _parfor_ParforFusionPass
from numba.parfors.parfor import ParforPreLoweringPass as \
    _parfor_ParforPreLoweringPass
from numba.parfors.parfor import Parfor
from numba.parfors.parfor_lowering import ParforLower

from numba.core.compiler_machinery import (FunctionPass, register_pass)
from numba.core.typed_passes import BaseNativeLowering, AnalysisPass


@register_pass(mutates_CFG=True, analysis_only=False)
class NativeParforLowering(BaseNativeLowering):
    """Lowering pass for a native function IR described using Numba's standard
    `numba.core.ir` nodes and also parfor.Parfor nodes."""
    _name = "native_parfor_lowering"

    @property
    def lowering_class(self):
        return ParforLower


@register_pass(mutates_CFG=True, analysis_only=False)
class PreParforPass(FunctionPass):

    _name = "pre_parfor_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        Preprocessing for data-parallel computations.
        """
        # Ensure we have an IR and type information.
        assert state.func_ir
        preparfor_pass = _parfor_PreParforPass(
            state.func_ir,
            state.typemap,
            state.calltypes,
            state.typingctx,
            state.targetctx,
            state.flags.auto_parallel,
            state.parfor_diagnostics.replaced_fns
        )

        preparfor_pass.run()
        return True


# this is here so it pickles and for no other reason
def _reload_parfors():
    """Reloader for cached parfors
    """
    # Re-initialize the parallel backend when load from cache.
    from numba.np.ufunc.parallel import _launch_threads
    _launch_threads()


@register_pass(mutates_CFG=True, analysis_only=False)
class ParforPass(FunctionPass):

    _name = "parfor_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        Convert data-parallel computations into Parfor nodes
        """
        # Ensure we have an IR and type information.
        assert state.func_ir
        parfor_pass = _parfor_ParforPass(state.func_ir,
                                         state.typemap,
                                         state.calltypes,
                                         state.return_type,
                                         state.typingctx,
                                         state.targetctx,
                                         state.flags.auto_parallel,
                                         state.flags,
                                         state.metadata,
                                         state.parfor_diagnostics)
        parfor_pass.run()

        # check the parfor pass worked and warn if it didn't
        has_parfor = False
        for blk in state.func_ir.blocks.values():
            for stmnt in blk.body:
                if isinstance(stmnt, Parfor):
                    has_parfor = True
                    break
            else:
                continue
            break

        if not has_parfor:
            # parfor calls the compiler chain again with a string
            if not (config.DISABLE_PERFORMANCE_WARNINGS or
                    state.func_ir.loc.filename == '<string>'):
                url = ("https://numba.readthedocs.io/en/stable/user/"
                       "parallel.html#diagnostics")
                msg = ("\nThe keyword argument 'parallel=True' was specified "
                       "but no transformation for parallel execution was "
                       "possible.\n\nTo find out why, try turning on parallel "
                       "diagnostics, see %s for help." % url)
                warnings.warn(errors.NumbaPerformanceWarning(msg,
                                                             state.func_ir.loc))

        # Add reload function to initialize the parallel backend.
        state.reload_init.append(_reload_parfors)
        return True


@register_pass(mutates_CFG=True, analysis_only=False)
class ParforFusionPass(FunctionPass):

    _name = "parfor_fusion_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        Do fusion of parfor nodes.
        """
        # Ensure we have an IR and type information.
        assert state.func_ir
        parfor_pass = _parfor_ParforFusionPass(state.func_ir,
                                               state.typemap,
                                               state.calltypes,
                                               state.return_type,
                                               state.typingctx,
                                               state.targetctx,
                                               state.flags.auto_parallel,
                                               state.flags,
                                               state.metadata,
                                               state.parfor_diagnostics)
        parfor_pass.run()

        return True


@register_pass(mutates_CFG=True, analysis_only=False)
class ParforPreLoweringPass(FunctionPass):

    _name = "parfor_prelowering_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        Prepare parfors for lowering.
        """
        # Ensure we have an IR and type information.
        assert state.func_ir
        parfor_pass = _parfor_ParforPreLoweringPass(state.func_ir,
                                                    state.typemap,
                                                    state.calltypes,
                                                    state.return_type,
                                                    state.typingctx,
                                                    state.targetctx,
                                                    state.flags.auto_parallel,
                                                    state.flags,
                                                    state.metadata,
                                                    state.parfor_diagnostics)
        parfor_pass.run()

        return True


@register_pass(mutates_CFG=False, analysis_only=True)
class DumpParforDiagnostics(AnalysisPass):

    _name = "dump_parfor_diagnostics"

    def __init__(self):
        AnalysisPass.__init__(self)

    def run_pass(self, state):
        if state.flags.auto_parallel.enabled:
            if config.PARALLEL_DIAGNOSTICS:
                if state.parfor_diagnostics is not None:
                    state.parfor_diagnostics.dump(config.PARALLEL_DIAGNOSTICS)
                else:
                    raise RuntimeError("Diagnostics failed.")
        return True
