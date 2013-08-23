'''
Implements explicit casting functions
'''
import itertools
from . import types
import numbapro as nb


def caster_generator():
    exts = []
    datatypes = [nb.float32, nb.float64,
                 nb.complex64, nb.complex128,
                 nb.int8, nb.int16, nb.int32, nb.int64,
                 nb.uint8, nb.uint16, nb.uint32, nb.uint64]

    registered = set()
    for src, dst in itertools.product(datatypes, datatypes):

        nbobj = dst
        src = types.NUMBA_TYPE_MAP[src]
        dst = types.NUMBA_TYPE_MAP[dst]

        class Cast(object):
            function = nbobj, (src,), dst

            def generic_implement(self, context, args, argtys, retty):
                return argtys[0].llvm_cast(context.builder, args[0], retty)

        exts.append(Cast)

    return exts


extensions = caster_generator()

