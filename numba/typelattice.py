from __future__ import print_function, division, absolute_import
import itertools
from . import types as nt


type_lattice = {}


def connect(a, b, w=1.0):
    type_lattice[(a, b)] = w


def biconnect(a, b, w=1.0):
    connect(a, b, w)
    connect(b, a, -w)


def _init():
    biconnect(nt.uint8, nt.int8, w=1.1)
    biconnect(nt.uint16, nt.int16, w=1.1)
    biconnect(nt.uint32, nt.int32, w=1.1)
    biconnect(nt.uint64, nt.int64, w=1.1)

    biconnect(nt.int8, nt.int16)
    biconnect(nt.int16, nt.int32)
    biconnect(nt.int32, nt.int64)
    biconnect(nt.int64, nt.float32, w=1.2)
    biconnect(nt.float32, nt.float64)

    connect(nt.float32, nt.complex64, w=1.1)
    connect(nt.float64, nt.complex128, w=1.1)
    connect(nt.float64, nt.complex64, w=-1.0)

    biconnect(nt.complex64, nt.complex128)

    # Identity
    for ty in nt.number_domain:
        connect(ty, ty, w=0)

    pending = []

    fullyconnected = nt.integer_domain|nt.real_domain

    for key in itertools.product(fullyconnected, fullyconnected):
        if key not in type_lattice:
            pending.append(key)

    for key in itertools.product(fullyconnected, nt.complex_domain):
        if key not in type_lattice:
            pending.append(key)


    while pending:
        a, b = pending.pop()

        for x in fullyconnected:
            first = (a, x)
            second = (x, b)
            if first in type_lattice and second in type_lattice:
                connect(a, b, w=(type_lattice[first] + type_lattice[second]))
                break
        else:
            pending = [(a, b)] + pending

_init()
