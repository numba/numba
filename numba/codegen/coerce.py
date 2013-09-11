# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import llvm

from numba import *
from numba import string_ as c_string_type
from numba import nodes
from numba.typesystem import is_obj, promote_to_native
from numba.codegen.codeutils import llvm_alloca, if_badval
from numba.codegen import debug


class ObjectCoercer(object):
    """
    Object that knows how to convert to/from objects using Py_BuildValue
    and PyArg_ParseTuple.
    """

    # TODO: do all of this in a specializer

    type_to_buildvalue_str = {
        char: "c",
        short: "h",
        int_: "i",
        long_: "l",
        longlong: "L",
        Py_ssize_t: "n",
        npy_intp: "n", # ?
        size_t: "n", # ?
        uchar: "B",
        ushort: "H",
        uint: "I",
        ulong: "k",
        ulonglong: "K",

        float_: "f",
        double: "d",
        complex128: "D",

        object_: "O",
        bool_: "b", # ?
        char.pointer(): "s",
        char.pointer() : "s",
        c_string_type: "s",
    }

    def __init__(self, translator):
        self.context = translator.context
        self.translator = translator
        self.builder = translator.builder
        self.llvm_module = self.builder.basic_block.function.module
        sig, self.py_buildvalue = self.context.external_library.declare(
            self.llvm_module, 'Py_BuildValue')
        sig, self.pyarg_parsetuple = self.context.external_library.declare(
            self.llvm_module, 'PyArg_ParseTuple')
        sig, self.pyerr_clear = self.context.external_library.declare(
            self.llvm_module, 'PyErr_Clear')
        self.function_cache = translator.function_cache
        self.NULL = self.translator.visit(nodes.NULL_obj)

    def check_err(self, llvm_result, callback=None, cmp=llvm.core.ICMP_EQ,
                  pos_node=None):
        """
        Check for errors. If the result is NULL, and error should have been set
        Jumps to translator.error_label if an exception occurred.
        """
        assert llvm_result.type.kind == llvm.core.TYPE_POINTER, llvm_result.type
        int_result = self.translator.builder.ptrtoint(llvm_result,
                                                      llvm_types._intp)
        NULL = llvm.core.Constant.int(int_result.type, 0)

        if callback:
            if_badval(self.translator, int_result, NULL,
                      callback=callback or default_callback, cmp=cmp)
        else:
            test = self.builder.icmp(cmp, int_result, NULL)
            name = 'no_error'
            if hasattr(pos_node, 'lineno'):
                name = 'no_error_%s' % error.format_pos(pos_node).rstrip(": ")
            bb = self.translator.append_basic_block(name)
            self.builder.cbranch(test, self.translator.error_label, bb)
            self.builder.position_at_end(bb)

        return llvm_result

    def check_err_int(self, llvm_result, badval):
        llvm_badval = llvm.core.Constant.int(llvm_result.type, badval)
        if_badval(self.translator, llvm_result, llvm_badval,
                  callback=lambda b, *args: b.branch(self.translator.error_label))

    def _create_llvm_string(self, str):
        return self.translator.visit(nodes.ConstNode(str, char.pointer()))

    def lstr(self, types, fmt=None):
        "Get an llvm format string for the given types"
        typestrs = []
        result_types = []
        for type in types:
            if is_obj(type):
                type = object_
            elif type.is_int:
                type = promote_to_native(type)

            result_types.append(type)
            typestrs.append(self.type_to_buildvalue_str[type])

        str = "".join(typestrs)
        if fmt is not None:
            str = fmt % str

        if debug.debug_conversion:
            self.translator.puts("fmt: %s" % str)

        result = self._create_llvm_string(str)
        return result_types, result

    def buildvalue(self, types, *largs, **kwds):
        # The caller should check for errors using check_err or by wrapping
        # its node in an ObjectTempNode
        name = kwds.get('name', '')
        fmt = kwds.get('fmt', None)
        types, lstr = self.lstr(types, fmt)
        largs = (lstr,) + largs

        if debug.debug_conversion:
            self.translator.puts("building... %s" % name)
            # func_type = object_(*types).pointer()
        # py_buildvalue = self.builder.bitcast(
        #         self.py_buildvalue, func_type.to_llvm(self.context))
        py_buildvalue = self.py_buildvalue
        result = self.builder.call(py_buildvalue, largs, name=name)

        if debug.debug_conversion:
            self.translator.puts("done building... %s" % name)
            nodes.print_llvm(self.translator.env, object_, result)
            self.translator.puts("--------------------------")

        return result

    def npy_intp_to_py_ssize_t(self, llvm_result, type):
        return llvm_result, type

    def py_ssize_t_to_npy_intp(self, llvm_result, type):
        return llvm_result, type

    def convert_single_struct(self, llvm_result, type):
        types = []
        largs = []
        for i, (field_name, field_type) in enumerate(type.fields):
            types.extend((c_string_type, field_type))
            largs.append(self._create_llvm_string(field_name))
            struct_attr = self.builder.extract_value(llvm_result, i)
            largs.append(struct_attr)

        return self.buildvalue(types, *largs, name='struct', fmt="{%s}")

    def convert_single(self, type, llvm_result, name=''):
        "Generate code to convert an LLVM value to a Python object"
        llvm_result, type = self.npy_intp_to_py_ssize_t(llvm_result, type)
        if type.is_struct:
            return self.convert_single_struct(llvm_result, type)
        elif type.is_complex:
            # We have a Py_complex value, construct a Py_complex * temporary
            new_result = llvm_alloca(self.translator.lfunc, self.builder,
                                     llvm_result.type, name='complex_temp')
            self.builder.store(llvm_result, new_result)
            llvm_result = new_result

        return self.buildvalue([type], llvm_result, name=name)

    def build_tuple(self, types, llvm_values):
        "Build a tuple from a bunch of LLVM values"
        assert len(types) == len(llvm_values)
        return self.buildvalue(lstr, *llvm_values, name='tuple', fmt="(%s)")

    def build_list(self, types, llvm_values):
        "Build a tuple from a bunch of LLVM values"
        assert len(types) == len(llvm_values)
        return self.buildvalue(types, *llvm_values, name='list',  fmt="[%s]")

    def build_dict(self, key_types, value_types, llvm_keys, llvm_values):
        "Build a dict from a bunch of LLVM values"
        types = []
        largs = []
        for k, v, llvm_key, llvm_value in zip(key_types, value_types,
                                              llvm_keys, llvm_values):
            types.append(k)
            types.append(v)
            largs.append(llvm_key)
            largs.append(llvm_value)

        return self.buildvalue(types, *largs, name='dict', fmt="{%s}")

    def parse_tuple(self, lstr, llvm_tuple, types, name=''):
        "Unpack a Python tuple into typed llvm variables"
        lresults = []
        for i, type in enumerate(types):
            var = llvm_alloca(self.translator.lfunc, self.builder,
                              type.to_llvm(self.context),
                              name=name and "%s%d" % (name, i))
            lresults.append(var)

        largs = [llvm_tuple, lstr] + lresults

        if debug.debug_conversion:
            self.translator.puts("parsing tuple... %s" % (types,))
            nodes.print_llvm(self.translator.env, object_, llvm_tuple)

        parse_result = self.builder.call(self.pyarg_parsetuple, largs)
        self.check_err_int(parse_result, 0)

        # Some conversion functions don't reset the exception state...
        # self.builder.call(self.pyerr_clear, [])

        if debug.debug_conversion:
            self.translator.puts("successfully parsed tuple...")

        return [self.builder.load(result) for result in lresults]

    def to_native(self, type, llvm_tuple, name=''):
        "Generate code to convert a Python object to an LLVM value"
        types, lstr = self.lstr([type])
        lresult, = self.parse_tuple(lstr, llvm_tuple, [type], name=name)
        return lresult
