from numba import ir


class WithContext(object):
    """A dummy object for use as contextmanager.
    This can be used as a contextmanager.
    """
    def __enter__(self):
        pass

    def __exit__(self, typ, val, tb):
        pass

    def mutate_with_body(self, func_ir, blocks, blk_start, blk_end,
                         body_blocks, dispatcher_factory):
        """Mutate the *blocks* to implement this contextmanager.

        Parameters
        ----------
        func_ir : FunctionIR
        blocks : dict[ir.Block]
        blk_start, blk_end : int
            labels of the starting and ending block of the context-maanger.
        body_block: sequence[int]
            A sequence of ìnt`s representing labels of the with-body
        dispatcher_factory : callable
            A callable that takes a `FunctionIR` and returns a `Dispatcher`.
        """
        raise NotImplementedError


def _get_var_parent(name):
    """Get parent of the variable given its name
    """
    # If not a temprary variable
    if not name.startswith('$'):
        # Return the base component of the name
        return name.split('.', )[0]


def _clear_blocks(blocks, to_clear):
    """Remove keys in *to_clear* from *blocks*.
    """
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

        lifted_blks = {k: blocks[k] for k in body_blocks}
        _mutate_with_block_callee(lifted_blks, blk_start, blk_end,
                                  inputs, outputs)

        # XXX: transform body-blocks to return the output variables
        lifted_ir = func_ir.derive(
            blocks=lifted_blks,
            arg_names=tuple(inputs),
            arg_count=len(inputs),
            force_non_generator=True,
            )

        dispatcher = dispatcher_factory(lifted_ir)

        newblk = _mutate_with_block_caller(
            dispatcher, blocks, blk_start, blk_end, inputs, outputs,
            )

        blocks[blk_start] = newblk
        _clear_blocks(blocks, body_blocks)
        return dispatcher


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


def _mutate_with_block_caller(dispatcher, blocks, blk_start, blk_end,
                              inputs, outputs):
    """Mutate *blocks* for the caller of a with-context.

    Parameters
    ----------
    dispatcher : Dispatcher
    blocks : dict[ir.Block]
    blk_start, blk_end : int
        labels of the starting and ending block of the context-maanger.
    inputs: sequence[str]
        Input argument names
    outputs: sequence[str]
        Output argument names
    """
    sblk = blocks[blk_start]
    scope = sblk.scope
    loc = sblk.loc

    newblk = ir.Block(scope=scope, loc=loc)
    fn = ir.Const(value=dispatcher, loc=loc)
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


def _mutate_with_block_callee(blocks, blk_start, blk_end, inputs, outputs):
    """Mutate *blocks* for the callee of a with-context.

    Parameters
    ----------
    blocks : dict[ir.Block]
    blk_start, blk_end : int
        labels of the starting and ending block of the context-maanger.
    inputs: sequence[str]
        Input argument names
    outputs: sequence[str]
        Output argument names
    """
    head_blk = min(blocks)
    temp_blk = blocks[head_blk]
    scope = temp_blk.scope
    loc = temp_blk.loc

    def make_prologue():
        """
        Make a new block that unwraps the argument and jump to the loop entry.
        This block is the entry block of the function.
        """
        block = ir.Block(scope=scope, loc=loc)
        # load args
        args = [ir.Arg(name=k, index=i, loc=loc)
                for i, k in enumerate(inputs)]
        for aname, aval in zip(inputs, args):
            tmp = ir.Var(scope=scope, name=aname, loc=loc)
            block.append(ir.Assign(target=tmp, value=aval, loc=loc))
        # jump to loop entry
        block.append(ir.Jump(target=head_blk, loc=loc))
        return block

    def make_epilogue():
        """
        Make a new block to prepare the return values.
        This block is the last block of the function.
        """
        block = ir.Block(scope=scope, loc=loc)
        # prepare tuples to return
        vals = [scope.get_exact(name=name) for name in outputs]
        tupexpr = ir.Expr.build_tuple(items=vals, loc=loc)
        tup = scope.make_temp(loc=loc)
        block.append(ir.Assign(target=tup, value=tupexpr, loc=loc))
        # return
        block.append(ir.Return(value=tup, loc=loc))
        return block

    blocks[blk_start] = make_prologue()
    blocks[blk_end] = make_epilogue()
