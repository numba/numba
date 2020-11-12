from __future__ import print_function, division, absolute_import

from numba.core.compiler_machinery import PassManager

from numba.core.untyped_passes import (ExtractByteCode, TranslateByteCode, FixupArgs,
                                  IRProcessing, DeadBranchPrune,
                                  RewriteSemanticConstants, InlineClosureLikes,
                                  GenericRewrites, WithLifting,
                                  InlineInlinables, FindLiterallyCalls,
                                  MakeFunctionToJitFunction,
                                  CanonicalizeLoopExit, CanonicalizeLoopEntry,
                                  ReconstructSSA,
                                  LiteralUnroll)

from numba.core.typed_passes import (NopythonTypeInference, AnnotateTypes,
                                NopythonRewrites, PreParforPass, ParforPass,
                                DumpParforDiagnostics, IRLegalization,
                                InlineOverloads, PreLowerStripPhis)

from .dppl_passes import (
        DPPLConstantSizeStaticLocalMemoryPass,
        DPPLPreParforPass,
        DPPLParforPass,
        SpirvFriendlyLowering,
        DPPLAddNumpyOverloadPass,
        DPPLAddNumpyRemoveOverloadPass,
        DPPLNoPythonBackend
        )

class DPPLPassBuilder(object):
    """
    This is the DPPL pass builder to run Intel GPU/CPU specific
    code-generation and optimization passes. This pass builder does
    not offer objectmode and interpreted passes.
    """

    @staticmethod
    def default_numba_nopython_pipeline(state, pm):
        """Adds the default set of NUMBA passes to the pass manager
        """
        if state.func_ir is None:
            pm.add_pass(TranslateByteCode, "analyzing bytecode")
            pm.add_pass(FixupArgs, "fix up args")
        pm.add_pass(IRProcessing, "processing IR")
        pm.add_pass(WithLifting, "Handle with contexts")

        # this pass adds required logic to overload default implementation of
        # Numpy functions
        pm.add_pass(DPPLAddNumpyOverloadPass, "dppl add typing template for Numpy functions")

        # Add pass to ensure when users are allocating static
        # constant memory the size is a constant and can not
        # come from a closure variable
        pm.add_pass(DPPLConstantSizeStaticLocalMemoryPass, "dppl constant size for static local memory")

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

        if state.flags.enable_ssa:
            pm.add_pass(ReconstructSSA, "ssa")
        # typing
        pm.add_pass(NopythonTypeInference, "nopython frontend")
        pm.add_pass(AnnotateTypes, "annotate types")

        # strip phis
        pm.add_pass(PreLowerStripPhis, "remove phis nodes")

        # optimisation
        pm.add_pass(InlineOverloads, "inline overloaded functions")



    @staticmethod
    def define_nopython_pipeline(state, name='dppl_nopython'):
        """Returns an nopython mode pipeline based PassManager
        """
        pm = PassManager(name)
        DPPLPassBuilder.default_numba_nopython_pipeline(state, pm)

        # Intel GPU/CPU specific optimizations
        pm.add_pass(DPPLPreParforPass, "Preprocessing for parfors")
        if not state.flags.no_rewrites:
            pm.add_pass(NopythonRewrites, "nopython rewrites")
        pm.add_pass(DPPLParforPass, "convert to parfors")

        # legalise
        pm.add_pass(IRLegalization, "ensure IR is legal prior to lowering")

        # lower
        pm.add_pass(SpirvFriendlyLowering, "SPIRV-friendly lowering pass")
        pm.add_pass(DPPLNoPythonBackend, "nopython mode backend")
        pm.add_pass(DPPLAddNumpyRemoveOverloadPass, "dppl remove typing template for Numpy functions")
        pm.finalize()
        return pm
