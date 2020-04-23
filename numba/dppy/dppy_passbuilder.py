from __future__ import print_function, division, absolute_import

from numba.compiler_machinery import PassManager

from numba.untyped_passes import (ExtractByteCode, TranslateByteCode, FixupArgs,
                                  IRProcessing, DeadBranchPrune,
                                  RewriteSemanticConstants, InlineClosureLikes,
                                  GenericRewrites, WithLifting,
                                  InlineInlinables, FindLiterallyCalls,
                                  MakeFunctionToJitFunction,
                                  CanonicalizeLoopExit, CanonicalizeLoopEntry,
                                  LiteralUnroll)

from numba.typed_passes import (NopythonTypeInference, AnnotateTypes,
                                NopythonRewrites, PreParforPass, ParforPass,
                                DumpParforDiagnostics, IRLegalization,
                                InlineOverloads)

from .dppy_passes import (
        DPPyPreParforPass,
        DPPyParforPass,
        SpirvFriendlyLowering,
        DPPyNoPythonBackend,
        InlineParforVectorize
        )

class DPPyPassBuilder(object):
    """
    This is the DPPy pass builder to run Intel GPU/CPU specific
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

        # legalise
        pm.add_pass(IRLegalization, "ensure IR is legal prior to lowering")


    @staticmethod
    def define_nopython_pipeline(state, name='dppy_nopython'):
        """Returns an nopython mode pipeline based PassManager
        """
        pm = PassManager(name)
        DPPyPassBuilder.default_numba_nopython_pipeline(state, pm)


        # Intel GPU/CPU specific optimizations
        pm.add_pass(DPPyPreParforPass, "Preprocessing for parfors")
        if not state.flags.no_rewrites:
            pm.add_pass(NopythonRewrites, "nopython rewrites")
        pm.add_pass(DPPyParforPass, "convert to parfors")
        #pm.add_pass(InlineParforVectorize, "inline vectorize inside parfors ")

        # lower
        pm.add_pass(SpirvFriendlyLowering, "SPIRV-friendly lowering pass")
        pm.add_pass(DPPyNoPythonBackend, "nopython mode backend")
        pm.finalize()
        return pm
