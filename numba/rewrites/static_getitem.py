from numba import ir, errors
from . import register_rewrite, Rewrite


@register_rewrite('before-inference')
class RewriteConstGetitems(Rewrite):
    """
    Rewrite IR expressions of the kind `getitem(value=arr, index=$constXX)`
    where `$constXX` is a known constant as
    `static_getitem(value=arr, index=<constant value>)`.
    """

    def match(self, func_ir, block, typemap, calltypes):
        self.getitems = getitems = {}
        self.block = block
        # Detect all getitem expressions and find which ones can be
        # rewritten
        for expr in block.find_exprs(op='getitem'):
            if expr.op == 'getitem':
                try:
                    const = func_ir.infer_constant(expr.index)
                except errors.ConstantInferenceError:
                    continue
                getitems[expr] = const

        return len(getitems) > 0

    def apply(self):
        """
        Rewrite all matching getitems as static_getitems.
        """
        new_block = self.block.copy()
        new_block.clear()
        for inst in self.block.body:
            if isinstance(inst, ir.Assign):
                expr = inst.value
                if expr in self.getitems:
                    const = self.getitems[expr]
                    new_expr = ir.Expr.static_getitem(value=expr.value,
                                                      index=const,
                                                      index_var=expr.index,
                                                      loc=expr.loc)
                    inst = ir.Assign(value=new_expr, target=inst.target,
                                     loc=inst.loc)
            new_block.append(inst)
        return new_block


@register_rewrite('before-inference')
class RewriteConstSetitems(Rewrite):
    """
    Rewrite IR statements of the kind `setitem(target=arr, index=$constXX, ...)`
    where `$constXX` is a known constant as
    `static_setitem(target=arr, index=<constant value>, ...)`.
    """

    def match(self, func_ir, block, typemap, calltypes):
        self.setitems = setitems = {}
        self.block = block
        # Detect all setitem statements and find which ones can be
        # rewritten
        for inst in block.find_insts(ir.SetItem):
            try:
                const = func_ir.infer_constant(inst.index)
            except errors.ConstantInferenceError:
                continue
            setitems[inst] = const

        return len(setitems) > 0

    def apply(self):
        """
        Rewrite all matching setitems as static_setitems.
        """
        new_block = self.block.copy()
        new_block.clear()
        for inst in self.block.body:
            if inst in self.setitems:
                const = self.setitems[inst]
                new_inst = ir.StaticSetItem(inst.target, const,
                                            inst.index, inst.value, inst.loc)
                new_block.append(new_inst)
            else:
                new_block.append(inst)
        return new_block
