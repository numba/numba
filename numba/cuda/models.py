from numba.core.extending import register_model, models
from numba.core import types
from numba.cuda.types import Dim3


@register_model(Dim3)
class Dim3Model(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('x', types.int32),
            ('y', types.int32),
            ('z', types.int32)
        ]
        super().__init__(dmm, fe_type, members)
