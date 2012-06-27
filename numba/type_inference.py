import opcode

from .minivect import minierror
from . import translate, utils, _numba_types as _types

class Variable(object):
    """
    Variables placed on the stack. They allow an indirection
    so, that when used in an operation, the correct LLVM type can be inserted.
    """

    def __init__(self, type, is_local=False, is_global=False,
                 is_constant=False, name=None):
        self.name = name
        self.type = type
        self.is_local = is_local
        self.is_constant = is_constant
        self.is_global = is_global

    @classmethod
    def from_variable(cls, variable, **kwds):
        result = cls(variable.type)
        vars(result).update(kwds)
        return result

    def __repr__(self):
        args = []
        if self.is_local:
            args.append("is_local=True")
        if self.is_global:
            args.append("is_global=True")
        if self.is_constant:
            args.append("is_constant=True")
        if self.name:
            args.append(name)

        if args:
            extra_info = ", ".join(args)
        else:
            extra_info = ""

        return '<Variable(type=%s%s)>' % (self.type, extra_info)

class TypeInferer(translate.CodeIterator):
    def __init__(self, context, func, input_arguments):
        super(TypeInferer, self).__init__(context, func)
        # Name of locals -> type
        self.symtab = {}
        # Bytecode instruction (which must be an expression) -> Variable
        self.variables = {}
        self.stack = []

        self.init_globals()
        self.init_locals(input_arguments)

        self.return_variables = []
        self.return_type = None

    def infer_types(self):
        """
        Infer types for the function.
        """
        for i, op, arg in utils.itercode(self.costr):
            name = opcode.opname[op]
            method = getattr(self, 'op_' + name, None)
            if method is not None:
                method(i, op, arg)

        # todo: in case of unpromotable types, return object?
        self.return_type = self.return_variables[0].type
        for return_variable in self.return_variables[1:]:
            self.return_type = self.promote(self.return_type,
                                            return_variable.type)

    def init_globals(self):
        for global_name in self.names:
            self.symtab[global_name] = Variable(None, name=global_name)

    def init_locals(self, input_arguments):
        for i, arg in enumerate(input_arguments):
            type = self.type_from_pyval(arg)
            varname = self.varnames[i]
            self.symtab[varname] = Variable(type, is_local=True, name=varname)

        for varname in self.varnames[len(input_arguments):]:
            self.symtab[varname] = Variable(None, is_local=True, name=varname)

    def promote(self, v1, v2):
        return self.context.promote_types(v1.type, v2.type)

    def assert_assignable(self, dst_type, src_type):
        self.promote(dst_type, src_type)

    def getvar(self, arg):
        varname = self.varnames[arg]
        return self.symtab[varname]

    def type_from_pyval(self, pyval):
        return self.context.typemapper.from_python(pyval)

    def append(self, i, variable):
        self.variables[i] = variable
        self.stack.append(variable)

    def op_LOAD_FAST(self, i, op, arg):
        self.append(i, self.getvar(arg))

    def op_STORE_FAST(self, i, op, arg):
        oldvar = self.getvar(arg)
        newvar = self.stack.pop()
        if oldvar.type is None:
            oldvar.type = newvar.type
        else:
            if oldvar.type != newvar.type:
                self.assert_assignable(oldvar.type, newvar.type)
                oldvar.type = self.promote(oldvar, newvar)

        self.variables[i] = oldvar

    def op_LOAD_GLOBAL(self, i, op, arg):
        self.append(i, self.getvar(arg))

    def op_LOAD_CONST(self, i, op, arg):
        const = self.constants[arg]
        type = self.type_from_pyval(const)
        variable = Variable(type, is_constant=True)
        self.append(i, variable)

    def binop(self, i, op, arg):
        arg2 = self.stack.pop(-1)
        arg1 = self.stack.pop(-1)
        self.append(i, Variable(self.promote(arg1, arg2)))

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
        # Todo: you only want to do this parsing once, save the args
        # number of arguments is arg
        args = [self.stack[-i] for i in range(arg,0,-1)]
        if arg > 0:
            self.stack = self.stack[:-arg]
        func = self.stack.pop()

        if func.is_global and func.name and func.name in ('range', 'xrange'):
            type = _types.IteratorType(minitypes.int_) # todo: Make this Py_ssize_t
            variable = Variable.from_variable(type, type=type)
        elif func.type.is_function:
            variable = Variable(func.type.return_type)
        elif func.type.is_object:
            variable = Variable(minitypes.object_)
        else:
            # TODO: implement
            raise NotImplementedError

        self.append(i, variable)

    def op_GET_ITER(self, i, op, arg):
        self.append(i, self.stack.pop())

    def op_FOR_ITER(self, i, op, arg):
        iterator = self.stack[-1].val
        if iterator.type.is_numba_type and iterator.type.is_iterator:
            self.append(i, Variable(iterator.type.base_type))
        else:
            raise NotImplementedError("Unknown type: %s" % (iterator.type,))

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
