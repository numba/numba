from numba import ir, ir_utils, types, errors, sigutils
from numba.typing.typeof import typeof_impl
from .transforms import find_region_inout_vars


class WithContext(object):
    """A dummy object for use as contextmanager.
    This can be used as a contextmanager.
    """
    is_callable = False

    def __enter__(self):
        pass

    def __exit__(self, typ, val, tb):
        pass

    def mutate_with_body(self, func_ir, blocks, blk_start, blk_end,
                         body_blocks, dispatcher_factory, extra):
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
                         body_blocks, dispatcher_factory, extra):
        assert extra is None
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
                         body_blocks, dispatcher_factory, extra):
        assert extra is None
        vlt = func_ir.variable_lifetime

        inputs, outputs = find_region_inout_vars(
            blocks=blocks,
            livemap=vlt.livemap,
            callfrom=blk_start,
            returnto=blk_end,
            body_block_ids=set(body_blocks),
            )

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


class _ObjModeContextType(WithContext):
    """Creates a contextmanager to be used inside jitted functions to enter
    *object-mode* for using interpreter features.  The body of the with-context
    is lifted into a function that is compiled into *object-mode*.  This
    transformation process is limitated and cannot process all possible
    Python code.  However, users can wrap complicated logic in another
    Python function, which will then be executed by the interpreter.

    Use this as a function that takes keyword arguments only.
    The argument names must correspond to the output variables from the
    with-block.  Their respective values are strings representing the expected
    types.  When exiting the with-context, the output variables are casted
    to the expected nopython types according to the annotation.  This process
    is the same as passing Python objects into arguments of a nopython
    function.

    Example::

        import numpy as np
        from numba import njit, objmode

        def bar(x):
            # This code is executed by the interpreter.
            return np.asarray(list(reversed(x.tolist()))

        @njit
        def foo():
            x = np.arange(5)
            with objmode(y='intp[:]'):  # annotate return type
                # this region is executed by object-mode.
                y += bar(x)
            return y

    .. note:: Known limitation:

        - with-block cannot use incoming list object.
        - with-block cannot use incoming function object.
        - with-block cannot ``yield``, ``break``, ``return`` or ``raise`` \
          such that the execution will leave the with-block immediately.
        - the ``objmode()`` function must be directly referenced as a global. \
          i.e. ``with numba.objmode():`` doesn't work.

    .. note:: When used outside of no-python mode, the context-manager has no
        effect.

    .. warning:: This feature is experimental.  The supported features may
        change with or without notice.

    """
    is_callable = True

    def _legalize_args(self, extra, loc):
        """
        Legalize arguments to the context-manager
        """
        if extra is None:
            return {}

        if len(extra['args']) != 0:
            raise errors.CompilerError(
                "objectmode context doesn't take any positional arguments",
                )
        callkwargs = extra['kwargs']
        typeanns = {}
        for k, v in callkwargs.items():
            if not isinstance(v, ir.Const) or not isinstance(v.value, str):
                raise errors.CompileError(
                    "objectmode context requires constants string for "
                    "type annotation",
                )

            typeanns[k] = sigutils._parse_signature_string(v.value)

        return typeanns

    def mutate_with_body(self, func_ir, blocks, blk_start, blk_end,
                         body_blocks, dispatcher_factory, extra):
        typeanns = self._legalize_args(extra, loc=blocks[blk_start].loc)
        vlt = func_ir.variable_lifetime

        inputs, outputs = find_region_inout_vars(
            blocks=blocks,
            livemap=vlt.livemap,
            callfrom=blk_start,
            returnto=blk_end,
            body_block_ids=set(body_blocks),
            )

        # Determine types in the output tuple
        def strip_var_ver(x):
            return x.split('.', 1)[0]

        stripped_outs = list(map(strip_var_ver, outputs))

        # Verify that only outputs are annotated
        extra_annotated = set(typeanns) - set(stripped_outs)
        if extra_annotated:
            msg = (
                'Invalid type annotation on non-outgoing variables: {}.'
                'Suggestion: remove annotation of the listed variables'
            )
            raise errors.TypingError(msg.format(extra_annotated))

        # Verify that all outputs are annotated
        not_annotated = set(stripped_outs) - set(typeanns)
        if not_annotated:
            msg = 'missing type annotation on outgoing variables: {}'
            raise errors.TypingError(msg.format(not_annotated))

        # Get output types
        outtup = types.Tuple([typeanns[v] for v in stripped_outs])

        lifted_blks = {k: blocks[k] for k in body_blocks}
        _mutate_with_block_callee(lifted_blks, blk_start, blk_end,
                                  inputs, outputs)

        lifted_ir = func_ir.derive(
            blocks=lifted_blks,
            arg_names=tuple(inputs),
            arg_count=len(inputs),
            force_non_generator=True,
            )

        dispatcher = dispatcher_factory(lifted_ir, objectmode=True,
                                        output_types=outtup)

        newblk = _mutate_with_block_caller(
            dispatcher, blocks, blk_start, blk_end, inputs, outputs,
            )

        blocks[blk_start] = newblk
        _clear_blocks(blocks, body_blocks)
        return dispatcher

    def __call__(self, *args, **kwargs):
        # No effect when used in pure-python
        return self


objmode_context = _ObjModeContextType()


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

