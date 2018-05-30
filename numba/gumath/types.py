from ndtypes import ndt
from xnd import xnd

from .. import types
from ..types.common import SimpleIterableType, SimpleIteratorType
from ..types.abstract import Type

from ..typing.typeof import typeof_impl

from ..datamodel import register_default
from ..datamodel.models import DataModel, StructModel

from .llvm import xnd_t


FROM_SCALAR = {
    ndt('bool'): types.boolean,
    ndt('int8'): types.int8,
    ndt('int16'): types.int16,
    ndt('int32'): types.int32,
    ndt('int64'): types.int64,

    ndt('uint8'): types.uint8,
    ndt('uint16'): types.uint16,
    ndt('uint32'): types.uint32,
    ndt('uint64'): types.uint64,

    ndt('float32'): types.float32,
    ndt('float64'): types.float64,

    ndt('complex64'): types.complex64,
    ndt('complex128'): types.complex128,
}

def type_from_scalar(x: xnd):
    return FROM_SCALAR[x.type.hidden_dtype]

class XNDIterator(SimpleIteratorType):
    def __init__(self, xnd_type): 
        self.xnd_type = xnd_type
        x = xnd_type.xnd
        super(XNDIterator, self).__init__(
            f'iter(xnd({x.type}))',
            XNDType(x[0]) if x.ndim > 1 else type_from_scalar(x)
        )


class XNDType(SimpleIterableType):
    def __init__(self, x: xnd):
        self.xnd = x
        super(XNDType, self).__init__(
            f'xnd({x.type})',
            XNDIterator(self)
        )

@typeof_impl.register(xnd)
def typeof_xnd(val, c):
    if val.ndim == 0:
        raise NotImplementedError('Scalar xnd inputs aren\'t implemented')
    return XNDType(val)


@register_default(XNDIterator)
class XNDIteratorModel(StructModel):
    def __init__(self, dmm, fe_type: XNDIterator):
        # We use an unsigned index to avoid the cost of negative index tests.
        members = [('index', types.EphemeralPointer(types.uintp)),
                   ('array', fe_type.xnd_type)]
        super(XNDIteratorModel, self).__init__(dmm, fe_type, members)



@register_default(XNDType)
class XNDModel(DataModel):
    def get_value_type(self):
        return xnd_t
    
    def from_argument(self, builder, val):
        return val
