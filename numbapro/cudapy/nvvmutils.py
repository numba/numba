from __future__ import absolute_import
import llvm.core as lc
from numba.cuda.cudadrv import nvvm


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
