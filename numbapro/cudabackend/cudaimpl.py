from __future__ import print_function, absolute_import, division
from llvm.core import Type
from numba.targets.imputils import implement, impl_attribute
from numba import cgutils
from numba import types


FUNCTIONS = []
ATTRIBUTES = []


def register(fn):
    FUNCTIONS.append(fn)


def register_attr(fn):
    ATTRIBUTES.append(fn)

# -----------------------------------------------------------------------------

SREG_MAPPING = {
    'tid.x': 'llvm.nvvm.read.ptx.sreg.tid.x',
    'tid.y': 'llvm.nvvm.read.ptx.sreg.tid.y',
    'tid.z': 'llvm.nvvm.read.ptx.sreg.tid.z',

    'ntid.x': 'llvm.nvvm.read.ptx.sreg.ntid.x',
    'ntid.y': 'llvm.nvvm.read.ptx.sreg.ntid.y',
    'ntid.z': 'llvm.nvvm.read.ptx.sreg.ntid.z',

    'ctaid.x': 'llvm.nvvm.read.ptx.sreg.ctaid.x',
    'ctaid.y': 'llvm.nvvm.read.ptx.sreg.ctaid.y',
    'ctaid.z': 'llvm.nvvm.read.ptx.sreg.ctaid.z',

    'nctaid.x': 'llvm.nvvm.read.ptx.sreg.nctaid.x',
    'nctaid.y': 'llvm.nvvm.read.ptx.sreg.nctaid.y',
    'nctaid.z': 'llvm.nvvm.read.ptx.sreg.nctaid.z',
}


def _call_sreg(builder, name):
    module = cgutils.get_module(builder)
    fnty = Type.function(Type.int(), ())
    fn = module.get_or_insert_function(fnty, name=SREG_MAPPING[name])
    return builder.call(fn, ())

# -----------------------------------------------------------------------------


@register
@implement('ptx.grid.1d', types.intp)
def ptx_grid1d(context, builder, sig, args):
    [ndim] = args
    tidx = _call_sreg(builder, "tid.x")
    ntidx = _call_sreg(builder, "ntid.x")
    nctaidx = _call_sreg(builder, "nctaid.x")

    res = builder.add(builder.mul(ntidx, nctaidx), tidx)
    return res


@register
@implement('tid.x')
def ptx_thread_idx(context, builder, sig, args):
    assert not args
    return _call_sreg(builder, "tid.x")
