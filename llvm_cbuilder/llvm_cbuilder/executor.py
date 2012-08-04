from llvm.core import Type, Module
import llvm.ee as le
import ctypes as ct

MAP_CTYPES = {
'void'       :   None,
'bool'       :   ct.c_bool,
'char'       :   ct.c_char,
'uchar'      :   ct.c_ubyte,
'short'      :   ct.c_short,
'ushort'     :   ct.c_ushort,
'int'        :   ct.c_int,
'uint'       :   ct.c_uint,
'long'       :   ct.c_long,
'ulong'      :   ct.c_ulong,

'int8'       :   ct.c_int8,
'uint8'      :   ct.c_uint8,
'int16'      :   ct.c_int16,
'uint16'     :   ct.c_uint16,
'int32'      :   ct.c_int32,
'uint32'     :   ct.c_uint32,
'int64'      :   ct.c_int64,
'uint64'     :   ct.c_uint64,

'float'      :   ct.c_float,
'double'     :   ct.c_double,
'longdouble' :   ct.c_longdouble,
}

class CExecutor(object):
    def __init__(self, mod_or_engine):
        if isinstance(mod_or_engine, Module):
            self.engine = le.EngineBuilder.new(mod_or_engine).create()
        else:
            self.engine = mod_or_engine

    def get_ctype_function(self, fn, *typeinfo):
        if len(typeinfo)==1 and isinstance(typeinfo[0], str):
            types = [ MAP_CTYPES[s.strip()] for s in typeinfo[0].split(',') ]
            if not types:
                retty = None
                argtys = []
            else:
                retty = types[0]
                argtys = types[1:]
        else:
            retty = typeinfo[0]
            argtys = typeinfo[1:]

        prototype = ct.CFUNCTYPE(retty, *argtys)
        fnptr = self.engine.get_pointer_to_function(fn)
        return prototype(fnptr)

