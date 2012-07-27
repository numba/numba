"""
Run type inference on Python bytecode. This is now deprecated, please
see the ast_type_inference module.
"""

import opcode
import __builtin__

from .minivect import minierror, minitypes
from . import translate, utils, _numba_types as _types
from .symtab import Variable

class TypeInferer(translate.CodeIterator):
    """
    Type inference. Initialize with a minivect context, a Python function,
    and either runtime function argument values or their given types (as
    a FunctionType).
    """

    def __init__(self, context, func, func_signature=None):
        super(TypeInferer, self).__init__(context, func)
        # Name of locals -> type
        self.symtab = {}
        # Bytecode instruction (which must be an expression) -> Variable
        self.variables = {}
        self.stack = []

        self.func_signature = func_signature
        self.given_return_type = func_signature.return_type
        self.return_variables = []
        self.return_type = None

    def infer_types(self):
        """
        Infer types for the function.
        """
        self.init_globals()
        self.init_locals()

        for i, op, arg in utils.itercode(self.costr):
            name = opcode.opname[op]
            method = getattr(self, 'op_' + name, None)
            if method is not None:
                method(i, op, arg)

        # todo: in case of unpromotable types, return object?
        self.return_type = self.return_variables[0].type
        for return_variable in self.return_variables[1:]:
            self.return_type = self.promote_types(
                        self.return_type, return_variable.type)

        ret_type = self.func_signature.return_type
        if ret_type and ret_type != self.return_type:
            self.assert_assignable(ret_type, self.return_type)
            self.return_type = self.promote_types(ret_type, self.return_type)

        self.func_signature = minitypes.FunctionType(
                return_type=self.return_type, args=self.func_signature.args)

    def init_globals(self):
        for global_name in self.names:
            if (global_name in self.func.__globals__ or not
                    getattr(__builtin__, global_name, None)):
                # FIXME: analyse the bytecode of the entire module, to determine
                # overriding of builtins
                type = _types.GlobalType(name=global_name)
            else:
                if global_name in ('range', 'xrange'):
                    type = _types.RangeType()
                else:
                    type = _types.BuiltinType(name=global_name)

            self.symtab[global_name] = Variable(type, name=global_name)

    def init_locals(self):
        arg_types = self.func_signature.args
        for i, arg_type in enumerate(arg_types):
            varname = self.varnames[i]
            self.symtab[varname] = Variable(arg_type, is_local=True, name=varname)

        for varname in self.varnames[len(arg_types):]:
            self.symtab[varname] = Variable(None, is_local=True, name=varname)

    def promote_types(self, t1, t2):
        return self.context.promote_types(t1, t2)

    def promote(self, v1, v2):
        return self.promote_types(v1.type, v2.type)

    def assert_assignable(self, dst_type, src_type):
        self.promote_types(dst_type, src_type)

    def type_from_pyval(self, pyval):
        return self.context.typemapper.from_python(pyval)

    def append(self, i, variable):
        self.variables[i] = variable
        self.stack.append(variable)

    def op_LOAD_FAST(self, i, op, arg):
        self.append(i, self.getlocal(arg))

    def op_STORE_FAST(self, i, op, arg):
        oldvar = Variable.from_variable(self.getlocal(arg))
        newvar = self.stack.pop()
        if oldvar.type is None:
            oldvar.type = newvar.type
            self.getlocal(arg).type = newvar.type
        elif oldvar.type != newvar.type:
            self.assert_assignable(oldvar.type, newvar.type)
            oldvar.type = self.promote(oldvar, newvar)

        oldvar.state = newvar
        self.variables[i] = oldvar

    def op_LOAD_GLOBAL(self, i, op, arg):
        self.append(i, self.getglobal(arg))

    def op_LOAD_CONST(self, i, op, arg):
        const = self.constants[arg]
        type = self.type_from_pyval(const)
        variable = Variable(type, is_constant=True)
        self.append(i, variable)

    def binop(self, i, op, arg):
        arg2 = self.stack.pop(-1)
        arg1 = self.stack.pop(-1)
        variable = Variable(self.promote(arg1, arg2))
        variable.state = arg1, arg2
        self.append(i, variable)

    op_BINARY_ADD = binop
    op_INPLACE_ADD = binop
    op_BINARY_SUBTRACT = binop
    op_BINARY_MULTIPLY = binop
    op_INPLACE_MULTIPLY = binop
    op_BINARY_DIVIDE = binop
    #op_BINARY_FLOOR_DIVIDE = binop
    op_BINARY_MODULO = binop
    op_BINARY_POWER = binop
    op_COMPARE_OP = binop

    def op_BINARY_FLOOR_DIVIDE(self, i, op, arg):
        self.binop(i, op, arg)
        self.variables[i].type = minitypes.int_

    def op_RETURN_VALUE(self, i, op, arg):
        self.variables[i] = self.stack.pop()
        self.return_variables.append(self.variables[i])

    def op_CALL_FUNCTION(self, i, op, arg):
        # number of arguments is arg
        args = [self.stack[-idx] for idx in range(arg,0,-1)]
        if arg > 0:
            self.stack = self.stack[:-arg]
        func = self.stack.pop()

        if func.type.is_range:
            result = Variable(func.type)
        elif func.type.is_builtin and func.name == 'len':
            result = Variable(_types.Py_ssize_t)
        elif func.type.is_function:
            result = Variable(func.type.return_type)
        elif func.type.is_object:
            result = Variable(minitypes.object_)
        else:
            # TODO: implement
            raise NotImplementedError(func.type)

        result.state = func, args
        self.append(i, result)

    def op_GET_ITER(self, i, op, arg):
        self.append(i, self.stack.pop())

    def op_FOR_ITER(self, i, op, arg):
        iterator = self.stack[-1]
        if iterator.type.is_iterator:
            base_type = iterator.type.base_type
        elif iterator.type.is_array:
            base_type = iterator.type.dtype
        elif iterator.type.is_range:
            base_type = _types.int32 # todo: Make this Py_ssize_t
        else:
            raise NotImplementedError("Unknown type: %s" % (iterator.type,))

        variable = Variable(base_type)
        variable.state = iterator
        self.append(i, variable)

    def op_LOAD_ATTR(self, i, op, arg):
        raise NotImplementedError

    def op_UNPACK_SEQUENCE(self, i, op, arg):
        raise NotImplementedError

    def _get_index_type(self, array_type, index_type):
        if array_type.is_array:
            assert index_type.is_int_like
            return array_type.dtype
        elif array_type.is_pointer:
            assert index_type.is_int_like
            return array_type.base_type
        elif array_type.is_object:
            return minitypes.object_
        else:
            raise NotImplementedError

    def op_BINARY_SUBSCR(self, i, op, arg):
        index_var = self.stack.pop()
        arr_var = self.stack.pop()
        self.append(i, Variable(self._get_index_type(arr_var.type,
                                                     index_var.type)))

    def op_BUILD_TUPLE(self, i, op, arg):
        self.append(i, Variable(_types.tuple_))

    def op_STORE_SUBSCR(self, i, op, arg):
        index_var = self.stack.pop()
        arr_var = self.stack.pop()
        value = self.stack.pop()

        result_type = self._get_index_type(arr_var.type, index_var.type)
        self.assert_assignable(result_type, value.type)
        self.variables[i] = Variable(result_type)

    def op_POP_TOP(self, i, op, arg):
        self.variables[i] = self.stack.pop()

    def op_POP_JUMP_IF_FALSE(self, i, op, arg):
        self.variables[i] = self.stack.pop()
