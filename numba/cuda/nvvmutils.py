from __future__ import print_function, absolute_import, division
import itertools
import llvmlite.llvmpy.core as lc
from .cudadrv import nvvm
from numba import cgutils


def declare_atomic_cas_int32(lmod):
    fname = '___numba_cas_hack'
    fnty = lc.Type.function(lc.Type.int(32),
           (lc.Type.pointer(lc.Type.int(32)), lc.Type.int(32), lc.Type.int(32)))
    return lmod.get_or_insert_function(fnty, fname)


def declare_atomic_add_float32(lmod):
    fname = 'llvm.nvvm.atomic.load.add.f32.p0f32'
    fnty = lc.Type.function(lc.Type.float(),
        (lc.Type.pointer(lc.Type.float(), 0), lc.Type.float()))
    return lmod.get_or_insert_function(fnty, name=fname)


def declare_atomic_add_float64(lmod):
    fname = '___numba_atomic_double_add'
    fnty = lc.Type.function(lc.Type.double(),
        (lc.Type.pointer(lc.Type.double()), lc.Type.double()))
    return lmod.get_or_insert_function(fnty, fname)


def declare_atomic_max_float32(lmod):
    fname = '___numba_atomic_float_max'
    fnty = lc.Type.function(lc.Type.float(),
        (lc.Type.pointer(lc.Type.float()), lc.Type.float()))
    return lmod.get_or_insert_function(fnty, fname)


def declare_atomic_max_float64(lmod):
    fname = '___numba_atomic_double_max'
    fnty = lc.Type.function(lc.Type.double(),
        (lc.Type.pointer(lc.Type.double()), lc.Type.double()))
    return lmod.get_or_insert_function(fnty, fname)


def declare_atomic_min_float32(lmod):
    fname = '___numba_atomic_float_min'
    fnty = lc.Type.function(lc.Type.float(),
        (lc.Type.pointer(lc.Type.float()), lc.Type.float()))
    return lmod.get_or_insert_function(fnty, fname)


def declare_atomic_min_float64(lmod):
    fname = '___numba_atomic_double_min'
    fnty = lc.Type.function(lc.Type.double(),
        (lc.Type.pointer(lc.Type.double()), lc.Type.double()))
    return lmod.get_or_insert_function(fnty, fname)


def insert_addrspace_conv(lmod, elemtype, addrspace):
    addrspacename = {
        nvvm.ADDRSPACE_SHARED: 'shared',
        nvvm.ADDRSPACE_LOCAL: 'local',
        nvvm.ADDRSPACE_CONSTANT: 'constant',
    }[addrspace]
    tyname = str(elemtype)
    tyname = {'float': 'f32', 'double': 'f64'}.get(tyname, tyname)
    s2g_name_fmt = 'llvm.nvvm.ptr.' + addrspacename + '.to.gen.p0%s.p%d%s'
    s2g_name = s2g_name_fmt % (tyname, addrspace, tyname)
    elem_ptr_ty = lc.Type.pointer(elemtype)
    elem_ptr_ty_addrspace = lc.Type.pointer(elemtype, addrspace)
    s2g_fnty = lc.Type.function(elem_ptr_ty,
                                [elem_ptr_ty_addrspace])
    return lmod.get_or_insert_function(s2g_fnty, s2g_name)


def declare_string(builder, value):
    lmod = builder.basic_block.function.module
    cval = lc.Constant.stringz(value)
    gl = lmod.add_global_variable(cval.type, name="_str",
                                  addrspace=nvvm.ADDRSPACE_CONSTANT)
    gl.linkage = lc.LINKAGE_INTERNAL
    gl.global_constant = True
    gl.initializer = cval

    charty = lc.Type.int(8)
    constcharptrty = lc.Type.pointer(charty, nvvm.ADDRSPACE_CONSTANT)
    charptr = builder.bitcast(gl, constcharptrty)

    conv = insert_addrspace_conv(lmod, charty, nvvm.ADDRSPACE_CONSTANT)
    return builder.call(conv, [charptr])

def declare_vprint(lmod):
    voidptrty = lc.Type.pointer(lc.Type.int(8))
    # NOTE: the second argument to vprintf() points to the variable-length
    # array of arguments (after the format)
    vprintfty = lc.Type.function(lc.Type.int(), [voidptrty, voidptrty])
    vprintf = lmod.get_or_insert_function(vprintfty, "vprintf")
    return vprintf

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

    'warpsize': 'llvm.nvvm.read.ptx.sreg.warpsize',
    'laneid': 'llvm.nvvm.read.ptx.sreg.laneid',
}


def call_sreg(builder, name):
    module = builder.module
    fnty = lc.Type.function(lc.Type.int(), ())
    fn = module.get_or_insert_function(fnty, name=SREG_MAPPING[name])
    return builder.call(fn, ())


class SRegBuilder(object):
    def __init__(self, builder):
        self.builder = builder

    def tid(self, xyz):
        return call_sreg(self.builder, 'tid.%s' % xyz)

    def ctaid(self, xyz):
        return call_sreg(self.builder, 'ctaid.%s' % xyz)

    def ntid(self, xyz):
        return call_sreg(self.builder, 'ntid.%s' % xyz)

    def nctaid(self, xyz):
        return call_sreg(self.builder, 'nctaid.%s' % xyz)

    def getdim(self, xyz):
        tid = self.tid(xyz)
        ntid = self.ntid(xyz)
        nctaid = self.ctaid(xyz)
        res = self.builder.add(self.builder.mul(ntid, nctaid), tid)
        return res


def get_global_id(builder, dim):
    sreg = SRegBuilder(builder)
    it = (sreg.getdim(xyz) for xyz in 'xyz')
    seq = list(itertools.islice(it, None, dim))
    if dim == 1:
        return seq[0]
    else:
        return seq
