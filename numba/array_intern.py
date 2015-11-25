from __future__ import print_function, division, absolute_import
import numpy as np
from numba import cgutils, ir, types, rewrites
from numba.typing.templates import FunctionTemplate


@rewrites.register_rewrite
class InternArrayAlloc(rewrites.Rewrite):
    def __init__(self, *args, **kwargs):
        super(InternArrayAlloc, self).__init__(*args, **kwargs)
        self._block = None
        self._matches = None
        self._enabled = self.pipeline.targetctx.enable_nrt == False
        if self._enabled:
            # Install a lowering hook if we are using this rewrite.
            special_ops = self.pipeline.targetctx.special_ops
            if 'interned_alloc' not in special_ops:
                special_ops['interned_alloc'] = _lower_interned_alloc

    def match(self, label, block, typemap, calltypes):
        if not self._enabled:
            return False
        # Match only the first block
        if label == 0 and block.body and len(calltypes) > 1:
            self._matches = {}
            self._block = block
            for i, tar, expr in self._iter_call(block.body):
                fnty = typemap.get(expr.func.name)
                if isinstance(fnty, types.Function):
                    if (isinstance(fnty.key, type) and
                            issubclass(fnty.key, FunctionTemplate)):
                        if fnty.key.key == np.empty:
                            self._matches[i] = expr, typemap[tar.name]
            return bool(self._matches)
        return False

    def _iter_call(self, body):
        for i, stmt in enumerate(body):
            if isinstance(stmt, ir.Assign):
                if isinstance(stmt.value, ir.Expr):
                    expr = stmt.value
                    if expr.op == 'call':
                        yield i, stmt.target, expr

    def apply(self):
        newstmts = {}
        for i, (expr, ty) in self._matches.items():
            expr = ir.Expr(op="interned_alloc",
                           loc=expr.loc,
                           args=expr.args,
                           ty=ty)
            orig = self._block.body[i]
            newstmts[i] = ir.Assign(value=expr, target=orig.target,
                                    loc=orig.loc)
        repls = [newstmts.get(i, stmt)
                 for i, stmt in enumerate(self._block.body)]
        newblk = ir.Block(scope=self._block.scope, loc=self._block.loc)
        newblk.body.extend(repls)
        return newblk


def _lower_interned_alloc(lower, expr):
    if expr.ty.layout != 'C':
        raise NotImplementedError("only support C layout")
    context = lower.context
    builder = lower.builder

    cast_intp = lambda x, t: context.cast(builder, x, t, types.intp)

    elemcount = lower.loadvar(expr.args[0].name)
    elemcount_type = lower.typeof(expr.args[0].name)
    if isinstance(elemcount_type, types.Integer):
        shape_values = [cast_intp(elemcount, elemcount_type)]
    elif isinstance(elemcount_type, types.BaseTuple):
        dims = [cast_intp(v, t)
                for v, t in zip(cgutils.unpack_tuple(builder, elemcount),
                                elemcount_type)]
        elemcount = dims[0]
        for d in dims[1:]:
            elemcount = builder.mul(elemcount, d)

        shape_values = dims
    else:
        msg = "invalid type for shape argument: {0}".format(elemcount_type)
        raise TypeError(msg)

    lldtype = context.get_data_type(expr.ty.dtype)
    stackptr = builder.alloca(lldtype, size=elemcount)

    arycls = context.make_array(expr.ty)
    ary = arycls(context, builder)

    itemsize = context.get_constant(types.intp,
                                    context.get_abi_sizeof(lldtype))

    stride_values = [itemsize]
    for dim in reversed(shape_values[1:]):
        stride_values.insert(0, builder.mul(stride_values[0], dim))

    shape = cgutils.pack_array(builder, shape_values)
    strides = cgutils.pack_array(builder, stride_values)

    assert len(shape_values) == len(stride_values)
    assert len(shape_values) == expr.ty.ndim
    context.populate_array(ary,
                           data=stackptr,
                           shape=shape,
                           strides=strides,
                           itemsize=itemsize,
                           meminfo=None)

    return ary._getvalue()
