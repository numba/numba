import typing as pt
from numba.core import types

ClassSpecType = pt.Union[
    pt.Sequence[pt.Tuple[str, types.Type]], pt.OrderedDict[str, types.Type]
]
FuncSpecType = pt.Union[str, pt.Tuple[types.Type, ...], types.Type]
