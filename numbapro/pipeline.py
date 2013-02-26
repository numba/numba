import ast as ast_module
from functools import partial

from numba import pipeline
from numba import environment
from numba.pipeline import ComposedPipelineStage
from numba.environment import insert_stage, TypedProperty
from numba.control_flow import cfstats

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
# Prange
#----------------------------------------------------------------------------

class ExpandPrange(pipeline.PipelineStage):
    def transform(self, ast, env):
        transformer = self.make_specializer(prange.PrangeExpander, ast, env)
        return transformer.visit(ast)

class RewritePrangePrivates(pipeline.PipelineStage):
    def transform(self, ast, env):
        transformer = self.make_specializer(prange.PrangePrivatesReplacer,
                                            ast, env)
        return transformer.visit(ast)

class UpdateAttributeStatements(pipeline.PipelineStage):
    def transform(self, ast, env):
        func_env = env.translation.crnt

        for block in func_env.flow.blocks:
            stats = []
            for cf_stat in block.stats:
                if isinstance(cf_stat, cfstats.AttributeAssignment):
                    value = cf_stat.lhs.value
                    if (isinstance(value, ast_module.Name) and
                            value.id in func_env.kill_attribute_assignments):
                        cf_stat = None

                if cf_stat:
                    stats.append(cf_stat)

            block.stats = stats

        return ast

class CleanupPrange(pipeline.PipelineStage):
    def transform(self, ast, env):
        transformer = self.make_specializer(prange.PrangeCleanup, ast, env)
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

insert_stage(order, ExpandPrange, before='ControlFlowAnalysis')
insert_stage(order, RewritePrangePrivates, before='ControlFlowAnalysis')
insert_stage(order, 'FixASTLocations', before='ControlFlowAnalysis')
insert_stage(order, UpdateAttributeStatements, before='TypeInfer')
insert_stage(order, CleanupPrange, after='TypeInfer')

idx = order.index('TypeInfer') + 1
type_infer_order = order[:idx]
compile_order = order[idx:]


#print order

#----------------------------------------------------------------------------
# Create Environment
#----------------------------------------------------------------------------

class NumbaproFunctionEnvironment(environment.FunctionEnvironment):
    """
    FunctionEnvironment for NumbaPro.
    """

    kill_attribute_assignments = TypedProperty(
        (set, frozenset),
        "Assignments to attributes that need to be removed from type "
        "inference pre-analysis. We need to do this for prange since we "
        "need to infer the types of variables to build a struct type for "
        "those variables. So we need type inference to be properly ordered, "
        "and not look at the attributes first.")

    def init(self, *args, **kws):
        super(NumbaproFunctionEnvironment, self).init(*args, **kws)
        self.kill_attribute_assignments = set()

    def getstate(self):
        state = super(NumbaproFunctionEnvironment, self).getstate()
        state.update(kill_attribute_assignments=self.kill_attribute_assignments)
        return state


class NumbaproEnvironment(environment.NumbaEnvironment):
    """
    Global numbapro environment.
    """

    FunctionEnvironment = NumbaproFunctionEnvironment

    def __init__(self, name, *args, **kws):
        self.default_pipeline = 'numbapro'
        super(NumbaproEnvironment, self).__init__(name, *args, **kws)

        self.pipelines.update({
            self.default_pipeline : pipeline.ComposedPipelineStage(order),
            'type_infer' : pipeline.ComposedPipelineStage(type_infer_order),
            'compile' : pipeline.ComposedPipelineStage(compile_order),
        })


numba_env = environment.NumbaEnvironment.get_environment()
numbapro_env = NumbaproEnvironment('numbapro')
numbapro_env.context.cbuilder_library = numba_env.context.cbuilder_library
env = numbapro_env
