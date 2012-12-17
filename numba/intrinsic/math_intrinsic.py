from intrinsic import Intrinsic, Signature
from numba import naming
from llpython.byte_translator import LLVMTranslator
import llvm.core

__all__ = ['PyModulo']

class PyModulo(Intrinsic):
    '''
    This is a generic definition that does not define arg_types and return_type.
    '''
    @property
    def name(self):
        return naming.specialized_mangle('__py_modulo', self.arg_types)

    def implementation(self, module, lfunc):
        _rtype = lfunc.type.pointee.return_type
        _rem = (llvm.core.Builder.srem
                if _rtype.kind == llvm.core.TYPE_INTEGER
                else llvm.core.Builder.frem)
        def _py_modulo (arg1, arg2):
            lfunc = rem(arg1, arg2)
            if lfunc < rtype(0):
                if arg2 > rtype(0):
                    lfunc += arg2
            elif arg2 < rtype(0):
                lfunc += arg2
            return lfunc
        LLVMTranslator(module).translate(_py_modulo, llvm_function = lfunc,
                                         env = {'rtype' : _rtype, 'rem' : _rem})
        return lfunc


