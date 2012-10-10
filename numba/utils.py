import opcode

from .minivect.complex_support import Complex64, Complex128, Complex256
from .minivect import miniast, minitypes
from . import _numba_types

def itercode(code):
    """Return a generator of byte-offset, opcode, and argument
    from a byte-code-string
    """
    i = 0
    extended_arg = 0
    n = len(code)
    while i < n:
        c = code[i]
        num = i
        op = ord(c)
        i = i + 1
        oparg = None
        if op >= opcode.HAVE_ARGUMENT:
            oparg = ord(code[i]) + ord(code[i + 1]) * 256 + extended_arg
            extended_arg = 0
            i = i + 2
            if op == opcode.EXTENDED_ARG:
                extended_arg = oparg * 65536L

        delta = yield num, op, oparg
        if delta is not None:
            abs_rel, dst = delta
            assert abs_rel == 'abs' or abs_rel == 'rel'
            i = dst if abs_rel == 'abs' else i + dst

def debugout(*args):
    '''This is a magic function.  If you use it in compiled functions,
    Numba should generate code for displaying the received value.'''
    if __debug__:
        print("debugout (non-translated): %s" % (''.join((str(arg)
                                                          for arg in args)),))


class NumbaContext(miniast.LLVMContext):
    # debug = True
    # debug_elements = True

    # Accept dynamic arguments
    astbuilder_cls = miniast.DynamicArgumentASTBuilder

    shape_type = minitypes.npy_intp.pointer()
    strides_type = shape_type

    def init(self):
        self.astbuilder = self.astbuilder_cls(self)
        self.typemapper = _numba_types.NumbaTypeMapper(self)

def get_minivect_context():
    return NumbaContext()

context = get_minivect_context()
