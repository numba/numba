from collections import namedtuple

from numba import environment as _env
from numba import pipeline as _nb_pipeline
from numba.utils import WriteOnceTypedProperty
from numba import environment as numba_env

import logging
logger = logging.getLogger(__name__)

PipelineOrders = namedtuple('PipelineOrders', ['default', 'type_infer'], )

def get_pipeline_orders():
    order = numba_env.default_pipeline_order[:]
    # Add constantfolding
    order.insert(order.index('ControlFlowAnalysis'), 'ConstFolding')
    # Remove linking stage because we are not putting it into the global module
    order.remove('LinkingStage')
    # Remove closure support
    order.remove('ClosureTypeInference')
    order.remove('SpecializeClosures')
    # Remove wrapping
    order.remove('WrapperStage')

    # We never use this
    type_infer_order = [] # order[:order.index('TypeInfer') + 1]
    return PipelineOrders(default=order, type_infer=type_infer_order)



class CUEnvironment(_env.NumbaEnvironment):
    def __init__(self, name, *args, **kws):
        super(CUEnvironment, self).__init__(name, *args, **kws)
        # add cbuilder_library
        assert not hasattr(self.context, 'cbuilder_library')
        numba_env = _env.NumbaEnvironment.get_environment()
        self.context.cbuilder_library = numba_env.context.cbuilder_library
        # setup pipeline
        orders = get_pipeline_orders()
        composer = _nb_pipeline.ComposedPipelineStage
        self.pipelines.update({
            self.default_pipeline: composer(orders.default),
            'type_infer': composer(orders.type_infer)
        })


#
# May need the folloing hack to implement pymodulo later
#
#    import re
#    from llvm import core as _lc
#    from llvm import ee as _le
#    from llvm import passes as _lp
#
#
#    regex_py_modulo = re.compile('__numba_specialized_\d+___py_modulo')
#
#    def _hack_to_implement_pymodulo(module):
#        '''XXX: I should fix the linkage instead.
#            '''
#        for func in module.functions:
#            if regex_py_modulo.match(func.name):
#                assert func.is_declaration
#                func.add_attribute(_lc.ATTR_ALWAYS_INLINE)
#                bb = func.append_basic_block('entry')
#                b = _lc.Builder.new(bb)
#                if func.type.pointee.return_type.kind == _lc.TYPE_INTEGER:
#                    rem = b.srem
#                else:
#                    raise Exception("Does not support modulo of float-point number.")
#                b.ret(rem(*func.args))
#                del b
#                del bb
#
