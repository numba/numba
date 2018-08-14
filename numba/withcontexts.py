from numba import ir, ir_utils, types
from numba.typing.typeof import typeof_impl


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
            A sequence of int's representing labels of the with-body
        dispatcher_factory : callable
            A callable that takes a `FunctionIR` and returns a `Dispatcher`.
        """
        raise NotImplementedError


@typeof_impl.register(WithContext)
def typeof_contextmanager(val, c):
    return types.ContextManager(val)


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
    """A simple context-manager that tells the compiler to lift the body of the
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
    replaces the head block with a new block that jumps to the end.

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
    """Make a new block that calls into the lifeted with-context.

    Parameters
    ----------
    dispatcher : Dispatcher
    blocks : dict[ir.Block]
    blk_start, blk_end : int
        labels of the starting and ending block of the context-manager.
    inputs: sequence[str]
        Input variable names
    outputs: sequence[str]
        Output variable names
    """
    sblk = blocks[blk_start]
    scope = sblk.scope
    loc = sblk.loc
    newblock = ir.Block(scope=scope, loc=loc)

    ir_utils.fill_block_with_call(
        newblock=newblock,
        callee=dispatcher,
        label_next=blk_end,
        inputs=inputs,
        outputs=outputs,
        )
    return newblock


def _mutate_with_block_callee(blocks, blk_start, blk_end, inputs, outputs):
    """Mutate *blocks* for the callee of a with-context.

    Parameters
    ----------
    blocks : dict[ir.Block]
    blk_start, blk_end : int
        labels of the starting and ending block of the context-manager.
    inputs: sequence[str]
        Input variable names
    outputs: sequence[str]
        Output variable names
    """
    head_blk = min(blocks)
    temp_blk = blocks[head_blk]
    scope = temp_blk.scope
    loc = temp_blk.loc

    blocks[blk_start] = ir_utils.fill_callee_prologue(
        block=ir.Block(scope=scope, loc=loc),
        inputs=inputs,
        label_next=head_blk,
        )
    blocks[blk_end] = ir_utils.fill_callee_epilogue(
        block=ir.Block(scope=scope, loc=loc),
        outputs=outputs,
    )

