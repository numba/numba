from numbapro.npm.types import *
from numbapro.npm.typing import ScalarType
from numbapro.npm.symbolic import Const


def bind_scalar_constants(blocks, globals, intp):
    intp = ScalarType('i%d' % intp)

    numbers = (int, long, float, complex)
    for blk in blocks.itervalues():
        for inst in blk.body:

            if inst.kind == 'Global':
                try:
                    gv = globals[inst.args.name]
                except KeyError:
                    pass
                else:
                    if isinstance(gv, numbers):
                        inst.kind = 'Const'
                        inst.args = Const(gv)
                    elif isinstance(gv, tuple) and all(isinstance(x, numbers)
                                                       for x in gv):
                        inst.kind = 'Const'
                        inst.args = Const(gv)
