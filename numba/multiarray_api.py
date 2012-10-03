'''multiarray_api

Defines a utility class for generating LLVM code that retrieves values
out of the Numpy array C API PyCObject/capsule.
'''
# ______________________________________________________________________

import llvm.core as lc
import llvm.ee as le

from numpy.core.multiarray import _ARRAY_API

from .llvm_types import _int1, _int8, _int32, _int64, _intp, \
    _void_star, _void_star_star, \
    _numpy_struct, _numpy_array

from .scrape_multiarray_api import get_include, process_source

# ______________________________________________________________________

class MultiarrayAPI (object):
    _type_map = {
            'char' : _int8,
            'int' : _int32, # Based on mixed gcc/clang experiments,
                            # assuming sizeof(int) == 4 appears to
                            # hold true, even on 64-bit systems.
            'unsigned char' : _int8, # XXX Loses unsigneded-ness
            'unsigned int' : _int32, # XXX
            'void' : lc.Type.void(),
            'npy_bool' : _int1,
            'npy_intp' : _intp,
            'npy_uint32' : _int32, # XXX
            'PyArrayObject' : _numpy_struct,
            'double' : lc.Type.double(),
            'size_t' : _intp, # XXX
            'npy_int64' : _int64,
            'npy_datetime' : _int64, # npy_common.h
            'npy_timedelta' : _int64, # npy_common.h
        }

    @classmethod
    def non_fn_ty_to_llvm (cls, c_ty_str):
        npointer = c_ty_str.count('*')
        if npointer == 0:
            base_ty = c_ty_str
        else:
            base_ty = c_ty_str[:-npointer].strip()
        if base_ty == 'void' and npointer > 0:
            base_ty = _int8
        elif base_ty not in cls._type_map:
            if npointer > 0:
                base_ty = _int8 # Basically cast into void *
            else:
                base_ty = _int32 # Or an int.
        else:
            base_ty = cls._type_map[base_ty]
        ret_val = base_ty
        for _ in xrange(npointer):
            ret_val = lc.Type.pointer(ret_val)
        return ret_val

    @classmethod
    def c_ty_str_to_llvm (cls, c_ty_str):
        ty_str_fn_split = [substr.strip() for substr in c_ty_str.split('(*)')]
        ret_val = cls.non_fn_ty_to_llvm(ty_str_fn_split[0])
        if len(ty_str_fn_split) > 1:
            arg_ty_strs = ty_str_fn_split[1][1:-1].split(', ')
            if len(arg_ty_strs) == 1 and arg_ty_strs[0].strip() == 'void':
                arg_ty_strs = []
            arg_tys = [cls.non_fn_ty_to_llvm(arg_ty_str.strip())
                       for arg_ty_str in arg_ty_strs]
            ret_val = lc.Type.pointer(lc.Type.function(ret_val, arg_tys))
        return ret_val

    def _add_loader (self, symbol_name, symbol_index, symbol_type):
        def _load_symbol (module, builder):
            api = module.get_global_variable_named('PyArray_API')
            load_val = builder.load(
                builder.gep(
                    builder.load(api),
                    [lc.Constant.int(_int32, symbol_index)]))
            return builder.bitcast(load_val, symbol_type)
        fn_name = "load_" + symbol_name
        _load_symbol.__name__ = fn_name
        setattr(self, fn_name, _load_symbol)
        return _load_symbol

    def __init__ (self, include_source_path = None):
        if include_source_path is None:
            include_source_path = get_include()
        self.api_map = process_source(include_source_path)
        for symbol_name, (symbol_index, c_ty_str) in self.api_map.iteritems():
            symbol_type = self.c_ty_str_to_llvm(c_ty_str)
            self._add_loader(symbol_name, symbol_index, symbol_type)
            setattr(self, symbol_name + '_ty', symbol_type)
        self.api_addr = None

    def calculate_api_addr (self):
        '''Constructs a dummy LLVM module that only links to the
        Python C API function PyCObject_AsVoidPtr().  This method then
        uses a LLVM execution engine to extract the multiarray API
        address from the _ARRAY_API object.'''
        module = lc.Module.new('import_arrayish_mod')
        # FIXME: Add test to see if we should be using the capsule API
        # instead of PyCObject.
        fn_ty = lc.Type.function(_void_star, [_void_star])
        pycobj_avp = module.add_function(fn_ty, 'PyCObject_AsVoidPtr')
        ee = le.ExecutionEngine.new(module)
        pycobj = le.GenericValue.pointer(id(_ARRAY_API))
        ee.run_static_ctors()
        voidptr = ee.run_function(pycobj_avp, [pycobj])
        ret_val = self.api_addr = voidptr.as_pointer()
        return ret_val

    def set_PyArray_API (self, module):
        '''Adds PyArray_API as a global variable to the input LLVM module.'''
        if self.api_addr is None:
            self.calculate_api_addr()
        api = module.add_global_variable(_void_star_star, "PyArray_API")
        api.initializer = lc.Constant.inttoptr(lc.Constant.int(_intp,
                                                               self.api_addr),
                                               _void_star_star)
        api.linkage = lc.LINKAGE_INTERNAL

# ______________________________________________________________________
# End of multiarray_api.py
