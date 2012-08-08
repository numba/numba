from .builder import CExternal
import llvm.core as lc
from . import shortnames as types

class LibC(CExternal):
    printf = lc.Type.function(types.int, [types.char_p], True)
    # TODO a lot more to add

