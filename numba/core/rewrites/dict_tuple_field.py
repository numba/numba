"""Remove unnecessary dictionary value-field ownership.

    d[key][constant_index]  ->  _getitem_tuple_field(d, key, constant_index)

Only the direct lookup expression followed immediately by a built-in tuple
projection is rewritten, and only when the lookup result is used once.
No lookup is hoisted and no statement with possible side effects is crossed.
"""
from numba.core import ir, types
from numba.core.rewrites import Rewrite, register_rewrite


def _uses(inst):
    """Var names read by this statement, excluding an assignment target."""
    if isinstance(inst, ir.Assign):
        value = inst.value
        if not hasattr(value, 'list_vars'):
            return []
        return [v.name for v in value.list_vars()]
    return [v.name for v in inst.list_vars()] if hasattr(inst, 'list_vars') else []


def _safe_member(typ):
    """Use only core scalar / ndarray representations with standard NRT."""
    if type(typ) in (types.Tuple, types.UniTuple):
        return all(_safe_member(t) for t in typ.types)
    return (type(typ) in (types.Integer, types.Boolean, types.Float,
                          types.Complex, types.Array) or
            typ == types.unicode_type)


@register_rewrite('before-inference')
class RewriteDictTupleField(Rewrite):
    def __init__(self, state=None):
        self.state = state
        self._match = None

    def match(self, func_ir, block, typemap, calltypes):
        self._match = None
        # This registry is also run on IR for inlined closures, where there
        # may be no dispatcher signature or argument types yet.
        if self.state is None or self.state.args is None:
            return False
        # The direct typed-dictionary function argument is recognised without
        # needing a speculative type/effect analysis of aliases.
        args = {}
        counts = {}
        for b in func_ir.blocks.values():
            for stmt in b.body:
                if isinstance(stmt, ir.Assign):
                    target = stmt.target.name
                    counts[target] = counts.get(target, 0) + 1
                    if (isinstance(stmt.value, ir.Arg) and
                            stmt.value.index < len(self.state.args)):
                        args[target] = self.state.args[stmt.value.index]

        for i, stmt in enumerate(block.body):
            if not (isinstance(stmt, ir.Assign) and
                    isinstance(stmt.value, ir.Expr) and
                    stmt.value.op == 'getitem'):
                continue
            lookup = stmt.value
            d, key = lookup.value, lookup.index
            if not isinstance(d, ir.Var) or not isinstance(key, ir.Var):
                continue
            dty = args.get(d.name)
            if (not isinstance(dty, types.DictType) or counts[d.name] != 1 or
                    type(dty.value_type) not in (types.Tuple, types.UniTuple)):
                continue
            if not all(_safe_member(t) for t in dty.value_type.types):
                continue
            # Do not move lookup across anything effectful. The only accepted
            # intervening IR is a literal integer constant.
            j = i + 1
            lit = None
            index_var = None
            if j < len(block.body):
                const = block.body[j]
                if (isinstance(const, ir.Assign) and
                        isinstance(const.value, ir.Const) and
                        type(const.value.value) is int):
                    lit = const.value.value
                    index_var = const.target
                    j += 1
            if j >= len(block.body):
                continue
            projected = block.body[j]
            if not (isinstance(projected, ir.Assign) and
                    isinstance(projected.value, ir.Expr)):
                continue
            expr = projected.value
            if (not isinstance(expr.value, ir.Var) or
                    expr.value.name != stmt.target.name):
                continue
            if expr.op == 'static_getitem':
                if type(expr.index) is not int:
                    continue
                if lit is not None and lit != expr.index:
                    continue
                lit = expr.index
            elif expr.op == 'getitem':
                if (lit is None or not isinstance(expr.index, ir.Var) or
                        expr.index.name != index_var.name):
                    continue
            else:
                continue
            n = len(dty.value_type.types)
            if lit is None or not -n <= lit < n:
                continue
            # The full tuple must not escape through another use. The chosen
            # field may be an array; the intrinsic owns that one reference.
            if sum(_uses(s).count(stmt.target.name)
                   for b in func_ir.blocks.values() for s in b.body) != 1:
                continue
            self._match = (block, i, j, lit, index_var,
                           d, key, projected.target, stmt.loc)
            return True
        return False

    def apply(self):
        from numba.typed.dictobject import _getitem_tuple_field

        (block, lookup_pos, projection_pos, index, index_var,
         d, key, out, loc) = self._match
        newblock = block.copy()
        scope = block.scope
        fn = scope.redefine('$dict_selected_field', loc)
        setup = []
        # Define a new literal *before* the call. In the original expression
        # `d[key][2]`, the index constant can occur after `d[key]`; reusing
        # that variable would introduce an SSA use-before-definition.
        index_var = scope.redefine('$dict_selected_index', loc)
        setup.append(ir.Assign(ir.Const(index, loc), index_var, loc))
        setup.append(ir.Assign(ir.Global('_getitem_tuple_field',
                                         _getitem_tuple_field, loc), fn, loc))
        setup.append(ir.Assign(ir.Expr.call(fn, [d, key, index_var], (), loc),
                               out, loc))
        result = []
        for i, stmt in enumerate(block.body):
            if i == lookup_pos:
                result.extend(setup)
            elif i != projection_pos:
                result.append(stmt)
        newblock.body = result
        return newblock
