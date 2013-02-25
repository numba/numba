from numba import environment as _env
from numba import pipeline as _nb_pipeline

from . import pipeline as _cuda_pipeline

class CudaEnvironment(_env.NumbaEnvironment):
    def __init__(self, name, *args, **kws):
        super(CudaEnvironment, self).__init__(name, *args, **kws)

        # add cbuilder_library
        assert not hasattr(self.context, 'cbuilder_library')
        numba_env = _env.NumbaEnvironment.get_environment()
        self.context.cbuilder_library = numba_env.context.cbuilder_library
        # setup pipeline
        orders = _cuda_pipeline.get_orders()
        composer = _nb_pipeline.ComposedPipelineStage
        self.pipelines.update({
            self.default_pipeline: composer(orders.default),
            'type_infer': composer(orders.type_infer)
        })

#        self.pipelines.update({
#                              self.default_pipeline : pipeline.ComposedPipelineStage(order),
#                              'type_infer' : pipeline.ComposedPipelineStage(type_infer_order),
#                              })

