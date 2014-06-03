from __future__ import absolute_import
import llvm.core as lc


def declare_vprint(lmod):
    voidptrty = lc.Type.pointer(lc.Type.int(8))
    vprintfty = lc.Type.function(lc.Type.int(), [voidptrty, voidptrty])
    vprintf = lmod.get_or_insert_function(vprintfty, "vprintf")
    return vprintf

