from functools import partial

from numba import pipeline
from numba import environment
from numba.pipeline import ComposedPipelineStage
from numba.environment import insert_stage

from numbapro import array_expressions, array_slicing
from numbapro.parallel import prange


class NumbaproPipeline(pipeline.Pipeline):
    """
    Pipeline that supports NumbaPro specific functionality
    """

    def __init__(self, context, func, ast, func_signature, **kwargs):
        super(NumbaproPipeline, self).__init__(context, func, ast,
                                               func_signature, **kwargs)
        self.try_insert_specializer('rewrite_array_expressions',
                                    before='specialize')

        self.try_insert_specializer('expand_prange',
                                    before='cfg')
        self.try_insert_specializer('rewrite_prange_privates',
                                    before='cfg')
        self.try_insert_specializer('fix_ast_locations',
                                    before='cfg')
        self.try_insert_specializer('cleanup_prange',
                                    after='type_infer')

    def expand_prange(self, ast):
        transformer = self.make_specializer(prange.PrangeExpander, ast)
        return transformer.visit(ast)

    def rewrite_prange_privates(self, ast):
        transformer = self.make_specializer(prange.PrangePrivatesReplacer, ast)
        return transformer.visit(ast)

    def cleanup_prange(self, ast):
        transformer = self.make_specializer(prange.PrangeCleanup, ast)
        return transformer.visit(ast)

#----------------------------------------------------------------------------
# Array Expressions
#----------------------------------------------------------------------------

class RewriteArrayExpressions(pipeline.PipelineStage):
    def transform(self, ast, env):
        transformer = self.make_specializer(
            array_expressions.ArrayExpressionRewriteNative, ast)
        return transformer.visit(ast)

#----------------------------------------------------------------------------
# Prange
#----------------------------------------------------------------------------

class ExpandPrange(pipeline.PipelineStage):
    def transform(self, ast, env):
        transformer = self.make_specializer(prange.PrangeExpander, ast)
        return transformer.visit(ast)

class RewritePrangePrivates(pipeline.PipelineStage):
    def transform(self, ast, env):
        transformer = self.make_specializer(prange.PrangePrivatesReplacer, ast)
        return transformer.visit(ast)

class CleanupPrange(pipeline.PipelineStage):
    def transform(self, ast, env):
        transformer = self.make_specializer(prange.PrangeCleanup, ast)
        return transformer.visit(ast)

#----------------------------------------------------------------------------
# Code Generation
#----------------------------------------------------------------------------

class NumbaProCodegen(array_slicing.NativeSliceCodegenMixin):
    """
    Support native slicing code generation and prange.
    """

NumbaproPipeline.add_mixin('codegen', NumbaProCodegen, before=True)

#----------------------------------------------------------------------------
# Create Pipeline
#----------------------------------------------------------------------------

order = environment.default_pipeline_order[:]

insert_stage(order, RewriteArrayExpressions, before='Specialize')
insert_stage(order, ExpandPrange, before='ControlFlowAnalysis')
insert_stage(order, RewritePrangePrivates, before='ControlFlowAnalysis')
insert_stage(order, 'FixASTLocations', before='ControlFlowAnalysis')
insert_stage(order, CleanupPrange, after='TypeInfer')

#----------------------------------------------------------------------------
# Create Environment
#----------------------------------------------------------------------------

create_numbapro_pipeline = partial(ComposedPipelineStage, order)

numbapro_env = environment.NumbaEnvironment()
numbapro_env.get_or_add_pipeline('numbapro', create_numbapro_pipeline)
numbapro_env.default_pipeline = 'numbapro'

env = numbapro_env