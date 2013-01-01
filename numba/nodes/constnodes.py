from numba.nodes import *
import numba.nodes

class ConstNode(Node):
    """
    Wrap a constant.
    """

    _attributes = ['type', 'pyval']

    def __init__(self, pyval, type=None):
        if type is None:
            type = context.typemapper.from_python(pyval)

        # if pyval is not _NULL:
        #     assert not type.is_object

        self.variable = Variable(type, is_constant=True, constant_value=pyval)
        self.type = type
        self.pyval = pyval

    def value(self, translator):
        builder = translator.builder

        type = self.type
        ltype = type.to_llvm(context)

        constant = self.pyval

        if constant is _NULL:
            lvalue = llvm.core.Constant.null(type.to_llvm(context))
        elif type.is_float:
            lvalue = llvm.core.Constant.real(ltype, constant)
        elif type.is_int:
            lvalue = llvm.core.Constant.int(ltype, constant)
        elif type.is_complex:
            real = ConstNode(constant.real, type.base_type)
            imag = ConstNode(constant.imag, type.base_type)
            lvalue = llvm.core.Constant.struct([real.value(translator),
                                                imag.value(translator)])
        elif type.is_pointer:
            addr_int = translator.visit(ConstNode(self.pyval, type=Py_uintptr_t))
            lvalue = translator.builder.inttoptr(addr_int, ltype)
        elif type.is_object:
            raise NotImplementedError("Use ObjectInjectNode")
        elif type.is_c_string:
            lvalue = translate._LLVMModuleUtils.get_string_constant(
                                            translator.llvm_module, constant)
            type_char_p = typesystem.c_string_type.to_llvm(translator.context)
            lvalue = translator.builder.bitcast(lvalue, type_char_p)
        elif type.is_function:
            # lvalue = map_to_function(constant, type, self.mod)
            raise NotImplementedError
        else:
            raise NotImplementedError("Constant %s of type %s" %
                                                        (self.pyval, type))

        return lvalue

    def cast(self, dst_type):
        if dst_type.is_int:
            caster = int
        elif dst_type.is_float:
            caster = float
        elif dst_type.is_complex:
            caster = complex
        else:
            raise NotImplementedError(dst_type)

        try:
            self.pyval = caster(self.pyval)
        except ValueError:
            if dst_type.is_int and self.type.is_c_string:
                raise
            raise minierror.UnpromotableTypeError((dst_type, self.type))
        self.type = dst_type
        self.variable.type = dst_type

    def __repr__(self):
        return "const(%s, %s)" % (self.pyval, self.type)

_NULL = object()
NULL_obj = ConstNode(_NULL, object_)
NULL = ConstNode(_NULL, void.pointer())
