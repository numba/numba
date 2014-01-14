from __future__ import print_function, division, absolute_import
import itertools
from numba import types


class TypeLattice(object):
    def __init__(self):
        self.conns = {}
        self.blacklist = set()
        self.types = set()

    def connect(self, fromty, toty, weight=1.):
        key = fromty, toty
        self.conns[key] = weight
        self.types.add(fromty)
        self.types.add(toty)

    def forbids(self, fromty, toty):
        self.blacklist.add((fromty, toty))

    def _set(self, k, v):
        if k not in self.blacklist:
            self.conns[k] = v

    def build(self):
        alltypes = tuple(self.types)
        pending = []

        # initialize
        for k in itertools.product(alltypes, alltypes):
            a, b = k
            rk = b, a
            if a == b:
                self._set(k, 0)
            elif k not in self.conns:
                if rk in self.conns:
                    self._set(k, -self.conns[rk])
                else:
                    if k not in self.blacklist:
                        pending.append(k)

        # span first expansion
        while pending:
            before = len(pending)
            tried = []
            for k in pending:
                a, b = k
                rk = b, a
                for t in alltypes:
                    k1 = a, t
                    k2 = t, b
                    if k1 in self.conns and k2 in self.conns:
                        w = self.conns[k1] + self.conns[k2]
                        self._set(k, w)
                        self._set(rk, -w)
                        break
                else:
                    tried.append(k)

            pending = tried
            after = len(pending)
            assert after < before, ("Not making progress", pending)

        return self.conns


def _build_type_lattice():
    lattice = TypeLattice()
    # Write out all promotion rules
    # int
    lattice.connect(types.int8, types.int16)
    lattice.connect(types.int16, types.int32)
    lattice.connect(types.int32, types.int64)
    # uint
    lattice.connect(types.uint8, types.uint16)
    lattice.connect(types.uint16, types.uint32)
    lattice.connect(types.uint32, types.uint64)
    # uint -> int
    lattice.connect(types.uint32, types.int32, weight=1.5)
    lattice.connect(types.uint64, types.int64, weight=1.5)
    # real
    lattice.connect(types.float32, types.float64)
    # complex
    lattice.connect(types.complex64, types.complex128)
    # int -> real
    lattice.connect(types.int32, types.float32, weight=1.75)
    lattice.connect(types.int32, types.float64)
    lattice.connect(types.int64, types.float64, weight=1.75)
    # real -> complex
    lattice.connect(types.float32, types.complex64, weight=1.5)
    lattice.connect(types.float64, types.complex128, weight=1.5)
    # No dowcast from complex
    for cty in [types.complex64, types.complex128]:
        for ty in types.integer_domain | types.real_domain:
            lattice.forbids(cty, ty)

    return lattice.build()


type_lattice = _build_type_lattice()


