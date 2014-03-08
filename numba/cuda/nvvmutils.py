from __future__ import print_function, absolute_import, division
import llvm.core as lc
from .cudadrv import nvvm


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


def declare_vprint(lmod):
    voidptrty = lc.Type.pointer(lc.Type.int(8))
    vprintfty = lc.Type.function(lc.Type.int(), [voidptrty, voidptrty])
    vprintf = lmod.get_or_insert_function(vprintfty, "vprintf")
    return vprintf


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
