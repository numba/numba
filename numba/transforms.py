"""
Implement transformation on Numba IR
"""

from __future__ import absolute_import, print_function

from collections import namedtuple, defaultdict

from numba.analysis import compute_cfg_from_blocks, find_top_level_loops
from numba import ir, errors, ir_utils
from numba.analysis import compute_use_defs


def _extract_loop_lifting_candidates(cfg, blocks):
    """
    Returns a list of loops that are candidate for loop lifting
    """
    # check well-formed-ness of the loop
    def same_exit_point(loop):
        "all exits must point to the same location"
        outedges = set()
        for k in loop.exits:
            succs = set(x for x, _ in cfg.successors(k))
            if not succs:
                # If the exit point has no successor, it contains an return
                # statement, which is not handled by the looplifting code.
                # Thus, this loop is not a candidate.
                return False
            outedges |= succs
        return len(outedges) == 1

    def one_entry(loop):
        "there is one entry"
        return len(loop.entries) == 1

    def cannot_yield(loop):
        "cannot have yield inside the loop"
        insiders = set(loop.body) | set(loop.entries) | set(loop.exits)
        for blk in map(blocks.__getitem__, insiders):
            for inst in blk.body:
                if isinstance(inst, ir.Assign):
                    if isinstance(inst.value, ir.Yield):
                        return False
        return True

    return [loop for loop in find_top_level_loops(cfg)
            if same_exit_point(loop) and one_entry(loop) and cannot_yield(loop)]


def find_region_inout_vars(blocks, livemap, callfrom, returnto, body_block_ids):
    """Find input and output variables to a block region.
    """
    inputs = livemap[callfrom]
    outputs = livemap[returnto]

    # ensure live variables are actually used in the blocks, else remove,
    # saves having to create something valid to run through postproc
    # to achieve similar
    loopblocks = {}
    for k in body_block_ids:
        loopblocks[k] = blocks[k]

    used_vars = set()
    def_vars = set()
    defs = compute_use_defs(loopblocks)
    for vs in defs.usemap.values():
        used_vars |= vs
    for vs in defs.defmap.values():
        def_vars |= vs
    used_or_defined = used_vars | def_vars

    # note: sorted for stable ordering
    inputs = sorted(set(inputs) & used_or_defined)
    outputs = sorted(set(outputs) & used_or_defined & def_vars)
    return inputs, outputs


_loop_lift_info = namedtuple('loop_lift_info',
                             'loop,inputs,outputs,callfrom,returnto')


def _loop_lift_get_candidate_infos(cfg, blocks, livemap):
    """
    Returns information on looplifting candidates.
    """
    loops = _extract_loop_lifting_candidates(cfg, blocks)
    loopinfos = []
    for loop in loops:
        [callfrom] = loop.entries   # requirement checked earlier
        an_exit = next(iter(loop.exits))  # anyone of the exit block
        [(returnto, _)] = cfg.successors(an_exit)  # requirement checked earlier

        local_block_ids = set(loop.body) | set(loop.entries) | set(loop.exits)
        inputs, outputs = find_region_inout_vars(
            blocks=blocks,
            livemap=livemap,
            callfrom=callfrom,
            returnto=returnto,
            body_block_ids=local_block_ids,
        )

        lli = _loop_lift_info(loop=loop, inputs=inputs, outputs=outputs,
                              callfrom=callfrom, returnto=returnto)
        loopinfos.append(lli)

    return loopinfos


def _loop_lift_modify_call_block(liftedloop, block, inputs, outputs, returnto):
    """
    Transform calling block from top-level function to call the lifted loop.
    """
    scope = block.scope
    loc = block.loc
    blk = ir.Block(scope=scope, loc=loc)

    ir_utils.fill_block_with_call(
        newblock=blk,
        callee=liftedloop,
        label_next=returnto,
        inputs=inputs,
        outputs=outputs,
    )
    return blk


def _loop_lift_prepare_loop_func(loopinfo, blocks):
    """
    Inplace transform loop blocks for use as lifted loop.
    """
    entry_block = blocks[loopinfo.callfrom]
    scope = entry_block.scope
    loc = entry_block.loc

    # Lowering assumes the first block to be the one with the smallest offset
    firstblk = min(blocks) - 1
    blocks[firstblk] = ir_utils.fill_callee_prologue(
        block=ir.Block(scope=scope, loc=loc),
        inputs=loopinfo.inputs,
        label_next=loopinfo.callfrom,
    )
    blocks[loopinfo.returnto] = ir_utils.fill_callee_epilogue(
        block=ir.Block(scope=scope, loc=loc),
        outputs=loopinfo.outputs,
    )


def _loop_lift_modify_blocks(func_ir, loopinfo, blocks,
                             typingctx, targetctx, flags, locals):
    """
    Modify the block inplace to call to the lifted-loop.
    Returns a dictionary of blocks of the lifted-loop.
    """
    from numba.dispatcher import LiftedLoop

    # Copy loop blocks
    loop = loopinfo.loop
    loopblockkeys = set(loop.body) | set(loop.entries) | set(loop.exits)
    loopblocks = dict((k, blocks[k].copy()) for k in loopblockkeys)
    # Modify the loop blocks
    _loop_lift_prepare_loop_func(loopinfo, loopblocks)

    # Create a new IR for the lifted loop
    lifted_ir = func_ir.derive(blocks=loopblocks,
                               arg_names=tuple(loopinfo.inputs),
                               arg_count=len(loopinfo.inputs),
                               force_non_generator=True)
    liftedloop = LiftedLoop(lifted_ir,
                            typingctx, targetctx, flags, locals)

    # modify for calling into liftedloop
    callblock = _loop_lift_modify_call_block(liftedloop, blocks[loopinfo.callfrom],
                                             loopinfo.inputs, loopinfo.outputs,
                                             loopinfo.returnto)
    # remove blocks
    for k in loopblockkeys:
        del blocks[k]
    # update main interpreter callsite into the liftedloop
    blocks[loopinfo.callfrom] = callblock
    return liftedloop


def loop_lifting(func_ir, typingctx, targetctx, flags, locals):
    """
    Loop lifting transformation.

    Given a interpreter `func_ir` returns a 2 tuple of
    `(toplevel_interp, [loop0_interp, loop1_interp, ....])`
    """
    blocks = func_ir.blocks.copy()
    cfg = compute_cfg_from_blocks(blocks)
    loopinfos = _loop_lift_get_candidate_infos(cfg, blocks,
                                               func_ir.variable_lifetime.livemap)
    loops = []
    for loopinfo in loopinfos:
        lifted = _loop_lift_modify_blocks(func_ir, loopinfo, blocks,
                                          typingctx, targetctx, flags, locals)
        loops.append(lifted)

    # Make main IR
    main = func_ir.derive(blocks=blocks)

    return main, loops


def canonicalize_cfg_single_backedge(blocks):
    """
    Rewrite loops that have multiple backedges.
    """
    cfg = compute_cfg_from_blocks(blocks)
    newblocks = blocks.copy()

    def new_block_id():
        return max(newblocks.keys()) + 1

    def has_multiple_backedges(loop):
        count = 0
        for k in loop.body:
            blk = blocks[k]
            edges = blk.terminator.get_targets()
            # is a backedge?
            if loop.header in edges:
                count += 1
                if count > 1:
                    # early exit
                    return True
        return False

    def yield_loops_with_multiple_backedges():
        for lp in cfg.loops().values():
            if has_multiple_backedges(lp):
                yield lp

    def replace_target(term, src, dst):
        def replace(target):
            return (dst if target == src else target)

        if isinstance(term, ir.Branch):
            return ir.Branch(cond=term.cond,
                             truebr=replace(term.truebr),
                             falsebr=replace(term.falsebr),
                             loc=term.loc)
        elif isinstance(term, ir.Jump):
            return ir.Jump(target=replace(term.target), loc=term.loc)
        else:
            assert not term.get_targets()
            return term

    def rewrite_single_backedge(loop):
        """
        Add new tail block that gathers all the backedges
        """
        header = loop.header
        tailkey = new_block_id()
        for blkkey in loop.body:
            blk = newblocks[blkkey]
            if header in blk.terminator.get_targets():
                newblk = blk.copy()
                # rewrite backedge into jumps to new tail block
                newblk.body[-1] = replace_target(blk.terminator, header,
                                                 tailkey)
                newblocks[blkkey] = newblk
        # create new tail block
        entryblk = newblocks[header]
        tailblk = ir.Block(scope=entryblk.scope, loc=entryblk.loc)
        # add backedge
        tailblk.append(ir.Jump(target=header, loc=tailblk.loc))
        newblocks[tailkey] = tailblk

    for loop in yield_loops_with_multiple_backedges():
        rewrite_single_backedge(loop)

    return newblocks


def canonicalize_cfg(blocks):
    """
    Rewrite the given blocks to canonicalize the CFG.
    Returns a new dictionary of blocks.
    """
    return canonicalize_cfg_single_backedge(blocks)


def with_lifting(func_ir, typingctx, targetctx, flags, locals):
    """With-lifting transformation

    Rewrite the IR to extract all withs.
    Only the top-level withs are extracted.
    Returns the (the_new_ir, the_lifted_with_ir)
    """
    from numba import postproc

    def dispatcher_factory(func_ir, objectmode=False, **kwargs):
        from numba.dispatcher import LiftedWith, ObjModeLiftedWith

        myflags = flags.copy()
        if objectmode:
            # Lifted with-block cannot looplift
            myflags.enable_looplift = False
            # Lifted with-block uses object mode
            myflags.enable_pyobject = True
            cls = ObjModeLiftedWith
        else:
            cls = LiftedWith
        return cls(func_ir, typingctx, targetctx, myflags, locals, **kwargs)

    postproc.PostProcessor(func_ir).run()  # ensure we have variable lifetime
    assert func_ir.variable_lifetime
    vlt = func_ir.variable_lifetime
    blocks = func_ir.blocks.copy()
    # find where with-contexts regions are
    withs = find_setupwiths(blocks)
    cfg = vlt.cfg
    _legalize_withs_cfg(withs, cfg, blocks)
    # For each with-regions, mutate them according to
    # the kind of contextmanager
    sub_irs = []
    for (blk_start, blk_end) in withs:
        body_blocks = []
        for node in _cfg_nodes_in_region(cfg, blk_start, blk_end):
            body_blocks.append(node)

        _legalize_with_head(blocks[blk_start])
        # Find the contextmanager
        cmkind, extra = _get_with_contextmanager(func_ir, blocks, blk_start)
        # Mutate the body and get new IR
        sub = cmkind.mutate_with_body(func_ir, blocks, blk_start, blk_end,
                                      body_blocks, dispatcher_factory,
                                      extra)
        sub_irs.append(sub)
    if not sub_irs:
        # Unchanged
        new_ir = func_ir
    else:
        new_ir = func_ir.derive(blocks)
    return new_ir, sub_irs


def _get_with_contextmanager(func_ir, blocks, blk_start):
    """Get the global object used for the context manager
    """
    _illegal_cm_msg = "Illegal use of context-manager."

    def get_var_dfn(var):
        """Get the definition given a variable"""
        return func_ir.get_definition(var)

    def get_ctxmgr_obj(dfn):
        """Return the context-manager object and extra info.

        The extra contains the arguments if the context-manager is used
        as a call.
        """
        # If the contextmanager used as a Call
        if isinstance(dfn, ir.Expr) and dfn.op == 'call':
            args = [get_var_dfn(x) for x in dfn.args]
            kws = {k: get_var_dfn(v) for k, v in dfn.kws}
            extra = {'args': args, 'kwargs': kws}
            dfn = func_ir.get_definition(dfn.func)
        else:
            extra = None

        # Check the contextmanager object
        if isinstance(dfn, ir.Global):
            ctxobj = dfn.value
            if ctxobj is not ir.UNDEFINED:
                return ctxobj, extra
            raise errors.CompilerError(
                "Undefined variable used as context manager",
                loc=blocks[blk_start].loc,
                )

        raise errors.CompilerError(_illegal_cm_msg, loc=dfn.loc)

    # Scan the start of the with-region for the contextmanager
    for stmt in blocks[blk_start].body:
        if isinstance(stmt, ir.EnterWith):
            var_ref = stmt.contextmanager
            dfn = func_ir.get_definition(var_ref)
            ctxobj, extra = get_ctxmgr_obj(dfn)
            if not hasattr(ctxobj, 'mutate_with_body'):
                raise errors.CompilerError(
                    "Unsupported context manager in use",
                    loc=blocks[blk_start].loc,
                    )
            return ctxobj, extra
    # No contextmanager found?
    raise errors.CompilerError(
        "malformed with-context usage",
        loc=blocks[blk_start].loc,
        )


def _legalize_with_head(blk):
    """Given *blk*, the head block of the with-context, check that it doesn't
    do anything else.
    """
    counters = defaultdict(int)
    for stmt in blk.body:
        counters[type(stmt)] += 1

    if counters.pop(ir.EnterWith) != 1:
        raise errors.CompilerError(
            "with's head-block must have exactly 1 ENTER_WITH",
            loc=blk.loc,
            )
    if counters.pop(ir.Jump) != 1:
        raise errors.CompilerError(
            "with's head-block must have exactly 1 JUMP",
            loc=blk.loc,
            )
    # Can have any number of del
    counters.pop(ir.Del, None)
    # There MUST NOT be any other statements
    if counters:
        raise errors.CompilerError(
            "illegal statements in with's head-block",
            loc=blk.loc,
            )


def _cfg_nodes_in_region(cfg, region_begin, region_end):
    """Find the set of CFG nodes that are in the given region
    """
    region_nodes = set()
    stack = [region_begin]
    while stack:
        tos = stack.pop()
        succs, _ = zip(*cfg.successors(tos))
        nodes = set([node for node in succs
                     if node not in region_nodes and
                     node != region_end])
        stack.extend(nodes)
        region_nodes |= nodes

    return region_nodes


def _legalize_withs_cfg(withs, cfg, blocks):
    """Verify the CFG of the with-context(s).
    """
    doms = cfg.dominators()
    postdoms = cfg.post_dominators()

    # Verify that the with-context has no side-exits
    for s, e in withs:
        loc = blocks[s].loc
        if s not in doms[e]:
            # Not sure what condition can trigger this error.
            msg = "Entry of with-context not dominating the exit."
            raise errors.CompilerError(msg, loc=loc)
        if e not in postdoms[s]:
            msg = (
                "Does not support with-context that contain branches "
                "(i.e. break/return/raise) that can leave the with-context. "
                "Details: exit of with-context not post-dominating the entry. "
            )
            raise errors.CompilerError(msg, loc=loc)


def find_setupwiths(blocks):
    """Find all top-level with.

    Returns a list of ranges for the with-regions.
    """
    def find_ranges(blocks):
        for blk in blocks.values():
            for ew in blk.find_insts(ir.EnterWith):
                yield ew.begin, ew.end

    def previously_occurred(start, known_ranges):
        for a, b in known_ranges:
            if s >= a and s < b:
                return True
        return False

    known_ranges = []
    for s, e in sorted(find_ranges(blocks)):
        if not previously_occurred(s, known_ranges):
            if e not in blocks:
                # this's possible if there's an exit path in the with-block
                raise errors.CompilerError(
                    'unsupported controlflow due to return/raise '
                    'statements inside with block'
                    )
            assert s in blocks, 'starting offset is not a label'
            known_ranges.append((s, e))

    return known_ranges
