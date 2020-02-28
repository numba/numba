from collections import namedtuple

from numba.core import types, ir
from numba.core.ir_utils import mk_unique_var
from numba.core.typing import signature


_CallableNode = namedtuple("BoundFunc", ["func", "sig"])


class ParforLoweringBuilder:
    def __init__(self, lowerer, scope, loc):
        self._lowerer = lowerer
        self._scope = scope
        self._loc = loc

    @property
    def _context(self):
        return self._lowerer.context

    @property
    def _typingctx(self):
        return self._context.typing_context

    @property
    def _typemap(self):
        return self._lowerer.fndesc.typemap

    @property
    def _calltypes(self):
        return self._lowerer.fndesc.calltypes

    def bind_global_function(self, fobj, ftype, args, kws={}):
        loc = self._loc
        varname = f"{fobj.__name__}_func"
        gvname = f"{fobj.__name__}"

        func_sig = self._typingctx.resolve_function_type(ftype, args, kws)
        func_var = self.assign(
            rhs=ir.Global(gvname, fobj, loc=loc), typ=ftype, name=varname
        )
        return _CallableNode(func=func_var, sig=func_sig)

    def make_const_variable(self, cval, typ, name="pf_const") -> ir.Var:
        return self.assign(
            rhs=ir.Const(cval, loc=self._loc), typ=typ, name=name
        )

    def make_tuple_variable(self, varlist, name="pf_tuple") -> ir.Var:
        loc = self._loc
        vartys = [self._typemap[x.name] for x in varlist]
        tupty = types.Tuple.from_types(vartys)
        return self.assign(
            rhs=ir.Expr.build_tuple(varlist, loc), typ=tupty, name=name
        )

    def assign(self, rhs, typ, name="pf_assign") -> ir.Var:
        loc = self._loc
        var = ir.Var(self._scope, mk_unique_var(name), loc)
        self._typemap[var.name] = typ
        redshape_assign = ir.Assign(rhs, var, loc)
        self._lowerer.lower_inst(redshape_assign)
        return var

    def call(self, callable_node, args, kws={}) -> ir.Var:
        call = ir.Expr.call(callable_node.func, args, kws, loc=self._loc)
        self._calltypes[call] = callable_node.sig
        return call

    def setitem(self, obj, index, val) -> ir.SetItem:
        loc = self._loc
        tm = self._typemap
        setitem = ir.SetItem(obj, index, val, loc=loc)
        self._lowerer.fndesc.calltypes[setitem] = signature(
            types.none, tm[obj.name], tm[index.name], tm[val.name]
        )
        self._lowerer.lower_inst(setitem)
        return setitem
