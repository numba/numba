from __future__ import print_function, division, absolute_import
import itertools
from . import types as nt


type_lattice = {}


def connect(a, b, w=1.0):
    type_lattice[(a, b)] = w


def biconnect(a, b, w=1.0):
    connect(a, b, w)
    connect(b, a, -w)


def same(a, b, w=1.0):
    connect(a, b, w)
    connect(b, a, w)


def _complete(pending, typeset):
    while pending:
        a, b = pending.pop()

        for x in typeset:
            first = (a, x)
            second = (x, b)
            if first in type_lattice and second in type_lattice:
                w1 = type_lattice[first]
                w2 = type_lattice[second]
                if w1 < 0 and w2 < 0 or w1 >= 0 and w2 >= 0:
                    connect(a, b, w=w1 + w2)
                    break
                else:
                    print(first, second, w1, w2)
        else:
            pending = [(a, b)] + pending


def _init():

    # Identity
    for ty in nt.number_domain:
        connect(ty, ty, w=0)

    # Seed
    bits = [8, 16, 32, 64]

    # Unsigned to Signed
    for bx, by in itertools.product(bits, bits):
        bitdiff = abs(bits.index(by) - bits.index(bx))
        tx = getattr(nt, 'uint%d' % bx)
        ty = getattr(nt, 'int%d' % by)
        same(tx, ty, w=-1.1 - bitdiff)

    # Same signedness
    for bx, by in itertools.product(bits, bits):
        bitdiff = bits.index(by) - bits.index(bx)
        tx = getattr(nt, 'int%d' % bx)
        ty = getattr(nt, 'int%d' % by)

        biconnect(tx, ty, w=bitdiff)

        tx = getattr(nt, 'uint%d' % bx)
        ty = getattr(nt, 'uint%d' % by)

        biconnect(tx, ty, w=bitdiff)

    # Int to Real
    single = [8, 16]
    double = [8, 16, 32]
    for bx in single:
        ux = getattr(nt, 'uint%d' % bx)
        sx = getattr(nt, 'int%d' % bx)
        diff = 3 - single.index(bx)
        biconnect(ux, nt.float32, w=diff + 1.1)
        biconnect(sx, nt.float32, w=diff)

    for bx in double:
        ux = getattr(nt, 'uint%d' % bx)
        sx = getattr(nt, 'int%d' % bx)
        diff = 4 - double.index(bx)
        biconnect(ux, nt.float64, w=diff + 1.1)
        biconnect(sx, nt.float64, w=diff)

    connect(nt.uint32, nt.float32, w=5.7)
    connect(nt.uint32, nt.float64, w=5.5)
    connect(nt.uint64, nt.float32, w=5.3)
    connect(nt.uint64, nt.float64, w=5.1)

    connect(nt.int32, nt.float32, w=5.6)
    connect(nt.int32, nt.float64, w=5.4)
    connect(nt.int64, nt.float32, w=5.2)
    connect(nt.int64, nt.float64, w=5)

    # Real to Real
    biconnect(nt.float32, nt.float64)

    # Int to Complex
    for bx in [8, 16, 32, 64]:
        ux = getattr(nt, 'uint%d' % bx)
        sx = getattr(nt, 'int%d' % bx)
        connect(ux, nt.complex64, w=6.2)
        connect(sx, nt.complex64, w=6.2)
        connect(ux, nt.complex128, w=6)
        connect(sx, nt.complex128, w=6)

    # Real to Complex
    for ty in [nt.float32, nt.float64]:
        connect(ty, nt.complex64, w=5.2)
        connect(ty, nt.complex128, w=5)

    biconnect(nt.complex64, nt.complex128)

_init()
