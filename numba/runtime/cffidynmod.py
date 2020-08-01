# adapted from cffi module c/_cffi_backend.c,
# available at https://bitbucket.org/cffi/cffi/ cffi module is licensed under
# The MIT License
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import llvmlite.ir as ir
from numba.core import cgutils

# typedef struct _ctypedescr {
#   PyObject_VAR_HEAD

#   struct _ctypedescr *ct_itemdescr;  /* ptrs and arrays: the item type */
#   PyObject *ct_stuff;                /* structs: dict of the fields
#                                         arrays: ctypedescr of the ptr type
#                                         function: tuple(abi, ctres, ctargs..)
#                                         enum: pair {"name":x},{x:"name"}
#                                         ptrs: lazily, ctypedescr of array */
#   void *ct_extra;                    /* structs: first field (not a ref!)
#                                         function types: cif_description
#                                         primitives: prebuilt "cif" object */

#   PyObject *ct_weakreflist;    /* weakref support */

#   PyObject *ct_unique_key;    /* key in unique_cache (a string, but not
#                                    human-readable) */

#   Py_ssize_t ct_size;     /* size of instances, or -1 if unknown */
#   Py_ssize_t ct_length;   /* length of arrays, or -1 if unknown;
#                                or alignment of primitive and struct types;
#                                always -1 for pointers */
#   int ct_flags;           /* CT_xxx flags */

#   int ct_name_position;   /* index in ct_name of where to put a var name */
#   char ct_name[1];        /* string, e.g. "int *" for pointers to ints */
#} CTypeDescrObject;

cffi_type_descr_t = ir.global_context.get_identified_type("_cffi_type_descr_t")
cffi_type_descr_t.set_body(
    ir.ArrayType(cgutils.int8_t, 24),  # 24-byte PyObject_VAR_HEAD, 0
    cffi_type_descr_t.as_pointer(),  # ct_itemdescr, 1
    cgutils.voidptr_t,  # ct_stuff, 2
    cgutils.voidptr_t,  # ct_extra, 3
    cgutils.voidptr_t,  # ct_weakreflist, 4
    cgutils.voidptr_t,  # ct_uniquekey, 5
    cgutils.intp_t,  # ct_size, 6
    cgutils.intp_t,  # ct_length, 7
    cgutils.int32_t,  # ct_flags, 8
    cgutils.int32_t,  # ct_name_position, 9
    ir.ArrayType(cgutils.int8_t, 1),  # ct_name, 10
)

# typedef struct {
#     PyObject_HEAD
#     CTypeDescrObject *c_type;
#     char *c_data;
#     PyObject *c_weakreflist;
# } CDataObject;
cffi_cdata_t = ir.LiteralStructType(
    [
        ir.ArrayType(cgutils.int8_t, 16),  # 16-byte PyObject_HEAD, 0
        cffi_type_descr_t.as_pointer(),  # CTypeDescrObject* ctypes, 1
        cgutils.voidptr_t,  # cdata, 2
        cgutils.voidptr_t,  # PyObject *c_weakreflist, 3
    ]
)


primitive_signed = 0x001  # signed integer
primitive_unsigned = 0x002  # unsigned integer
primitive_char = 0x004  # char, wchar_t, charN_t
primitive_float = 0x008  # float, double, long double
primitive_complex = 0x400  # float _Complex, double _Complex

primitive_any = (
    primitive_signed
    | primitive_unsigned
    | primitive_char
    | primitive_float
    | primitive_complex
)


def get_cffi_pointer(builder, ptr):
    cffi_data_ptr = builder.bitcast(ptr, cffi_cdata_t.as_pointer())
    return builder.extract_value(builder.load(cffi_data_ptr), 2)


def get_cffi_value_type(builder, ptr):
    return builder.load(builder.gep(ptr,
                                    [
                                        cgutils.int32_t(0),
                                        cgutils.int32_t(1)
                                    ]))


def is_primitive_value(builder, ptr):
    typ = get_cffi_value_type(builder, ptr)
    flags = builder.extract_value(typ, 8)  # ct_flags
    return builder.icmp_unsigned(
        "!=", builder.and_(flags, cgutils.int32_t(
            primitive_any)), cgutils.int32_t(0)
    )
