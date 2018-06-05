from pprint import pprint

from numba import ir
from numba import transforms


class WithContext(object):
    pass


def _get_var_parent(name):
    if not name.startswith('$'):
        return name.split('.', )[0]


def _clear_blocks(blocks, to_clear):
    for b in to_clear:
        del blocks[b]


class _ByPassContextType(WithContext):
    """A simple context-manager that tells the compiler to bypass the body
    of the with-block.
    """
    def mutate_with_body(self, func_ir, blocks, blk_start, blk_end,
                         body_blocks, dispatcher_factory):
        # Determine variables that need forwarding
        vlt = func_ir.variable_lifetime
        inmap = {_get_var_parent(k): k for k in vlt.livemap[blk_start]}
        outmap = {_get_var_parent(k): k for k in vlt.livemap[blk_end]}
        forwardvars = {inmap[k]: outmap[k] for k in filter(bool, outmap)}
        # Transform the block
        _bypass_with_context(blocks, blk_start, blk_end, forwardvars)
        _clear_blocks(blocks, body_blocks)


bypass_context = _ByPassContextType()


class _CallContextType(WithContext):
    """A simple context-manager that tells the compiler lift the body of the
    with-block as another function.
    """
    def mutate_with_body(self, func_ir, blocks, blk_start, blk_end,
                         body_blocks, dispatcher_factory):
        vlt = func_ir.variable_lifetime
        inputs = vlt.livemap[blk_start]
        outputs = vlt.livemap[blk_end]

        # XXX: transform body-blocks to return the output variables

        lifted_ir = func_ir.derive(
            blocks={k: blocks[k] for k in body_blocks},
            arg_names=tuple(inputs),
            arg_count=len(inputs),
            force_non_generator=True,
            )

        newblk = _mutate_with_block_caller(
            lifted_ir, blocks, blk_start, blk_end, inputs, outputs,
            dispatcher_factory,
            )

        newblk.dump()
        raise


call_context = _CallContextType()


def _bypass_with_context(blocks, blk_start, blk_end, forwardvars):
    """Given the starting and ending block of the with-context,
    replaces a new head block that jumps to the end.

    *blocks* is modified inplace.
    """
    sblk = blocks[blk_start]
    scope = sblk.scope
    loc = sblk.loc
    newblk = ir.Block(scope=scope, loc=loc)
    for k, v in forwardvars.items():
        newblk.append(ir.Assign(value=scope.get_exact(k),
                                target=scope.get_exact(v),
                                loc=loc))
    newblk.append(ir.Jump(target=blk_end, loc=loc))
    blocks[blk_start] = newblk


def _mutate_with_block_caller(lifted_ir, blocks, blk_start, blk_end,
                              inputs, outputs, dispatcher_factory):
    # XXX: refactor this
    jitted_fn = dispatcher_factory(lifted_ir)

    sblk = blocks[blk_start]
    scope = sblk.scope
    loc = sblk.loc

    newblk = ir.Block(scope=scope, loc=loc)
    fn = ir.Const(value=jitted_fn, loc=loc)
    fnvar = scope.make_temp(loc=loc)
    newblk.append(ir.Assign(target=fnvar, value=fn, loc=loc))
    # call
    args = [scope.get_exact(name) for name in inputs]
    callexpr = ir.Expr.call(func=fnvar, args=args, kws=(), loc=loc)
    callres = scope.make_temp(loc=loc)
    newblk.append(ir.Assign(target=callres, value=callexpr, loc=loc))

    # unpack return value
    for i, out in enumerate(outputs):
        target = scope.get_exact(out)
        getitem = ir.Expr.static_getitem(value=callres, index=i,
                                         index_var=None, loc=loc)
        newblk.append(ir.Assign(target=target, value=getitem, loc=loc))
    # jump to next block
    newblk.append(ir.Jump(target=blk_end, loc=loc))
    return newblk
