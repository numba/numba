from collections import namedtuple
from numba import environment, pipeline as _pipeline
import logging
logger = logging.getLogger(__name__)

PipelineOrders = namedtuple('PipelineOrders', ['default', 'type_infer'], )

def get_orders():
    order = environment.default_pipeline_order[:]
    # Replace TypeInfer
    idx_typeinfer = order.index('TypeInfer')
    order[idx_typeinfer] = CudaTypeInferStage
    # Remove linking stage because we are not putting it into the global module
    order.remove('LinkingStage')
    # Remove closure support
    order.remove('ClosureTypeInference')
    order.remove('SpecializeClosures')
    # Replace CodeGen
    idx_codegen = order.index('CodeGen')
    order[idx_codegen] = CudaCodeGenStage
    # Remove wrapping
    order.remove('WrapperStage')

    # We never use this
    type_infer_order = [] # order[:order.index('TypeInfer') + 1]
    return PipelineOrders(default=order, type_infer=type_infer_order)

#
# Pipeline Stages
#
from . import transforms as _transforms

class CudaTypeInferStage(_pipeline.TypeInfer):
    def transform(self, ast, env):
        crnt = env.translation.crnt
        type_inferer = self.make_specializer(_transforms.CudaTypeInferer,
                                             ast, env, **crnt.kwargs)
        type_inferer.infer_types()
        crnt.func_signature = type_inferer.func_signature
        logger.debug("signature for %s: %s", crnt.func_name,
                     crnt.func_signature)
        crnt.symtab = type_inferer.symtab
        return ast


class CudaCodeGenStage(_pipeline.PipelineStage):
    def transform(self, ast, env):
        func_env = env.translation.crnt
        func_env.translator = self.make_specializer(
                                            _transforms.CudaCodeGenerator,
                                            ast, env, **func_env.kwargs)
        func_env.translator.translate()
        func_env.lfunc = func_env.translator.lfunc
        return ast
