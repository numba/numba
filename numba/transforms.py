"""
Implement transformation on Numba IR
"""

from __future__ import absolute_import, print_function

from collections import namedtuple

from numba.analysis import (compute_use_defs, compute_live_map,
                            compute_cfg_from_blocks, find_top_level_loops)
from numba import ir
from numba.interpreter import Interpreter


def _extract_loop_lifting_candidates(cfg, blocks):
    """
    Returns a list of loops that are candidate for loop lifting
    """
    # check well-formed-ness of the loop
    def same_exit_point(loop):
        "all exits must point to the same location"
        outedges = set()
        for k in loop.exits:
            outedges |= set(x for x, _ in cfg.successors(k))
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


_loop_lift_info = namedtuple('loop_lift_info',
                             'loop,inputs,outputs,callfrom,returnto')


def _loop_lift_get_candidate_infos(cfg, blocks):
    """
    Returns information on looplifting candidates.
    """
    loops = _extract_loop_lifting_candidates(cfg, blocks)

    usedefs = compute_use_defs(blocks)
    livemap = compute_live_map(cfg, blocks, usedefs.usemap, usedefs.defmap)

    loopinfos = []
    for loop in loops:
        [callfrom] = loop.entries   # requirement checked earlier
        an_exit = next(iter(loop.exits))
        [(returnto, _)] = cfg.successors(an_exit)  # requirement checked earlier

        # note: sorted for stable ordering
        inputs = sorted(livemap[callfrom])
        outputs = sorted(livemap[returnto])
        loopinfos.append(_loop_lift_info(loop=loop,
                                         inputs=inputs, outputs=outputs,
                                         callfrom=callfrom, returnto=returnto))

    return loopinfos


def _loop_lift_modify_call_block(liftedloop, block, inputs, outputs, returnto):
    """
    Transform calling block from top-level function to call the lifted loop.
    """
    scope = block.scope
    loc = block.loc
    blk = ir.Block(scope=scope, loc=loc)

    # XXX: should fix delete insertion
    # copy early deletes
    for inst in block.body:
        if isinstance(inst, ir.Del):
            blk.append(inst)
        else:
            break

    # load loop
    fn = ir.Const(value=liftedloop, loc=loc)
    fnvar = scope.make_temp(loc=loc)
    blk.append(ir.Assign(target=fnvar, value=fn, loc=loc))
    # call loop
    args = [scope.get(name) for name in inputs]
    callexpr = ir.Expr.call(func=fnvar, args=args, kws=(), loc=loc)

    callres = scope.make_temp(loc=loc)
    blk.append(ir.Assign(target=callres, value=callexpr, loc=loc))

    # unpack return value
    for i, out in enumerate(outputs):
        target = scope.get(out)
        getitem = ir.Expr.static_getitem(value=callres, index=i,
                                         index_var=None, loc=loc)
        blk.append(ir.Assign(target=target, value=getitem, loc=loc))

    # clean up
    blk.append(ir.Del(value=fnvar.name, loc=loc))
    blk.append(ir.Del(value=callres.name, loc=loc))

    # jump to next block
    blk.append(ir.Jump(target=returnto, loc=loc))
    return blk


def _loop_lift_prepare_loop_func(loopinfo, blocks):
    """
    Transform loop blocks for use as lifted loop
    """
    def make_prologue():
        entry_block = blocks[loopinfo.callfrom]
        scope = entry_block.scope
        loc = entry_block.loc

        block = ir.Block(scope=scope, loc=loc)
        # load args
        args = [ir.Arg(name=k, index=i, loc=loc)
                for i, k in enumerate(loopinfo.inputs)]
        for aname, aval in zip(loopinfo.inputs, args):
            tmp = ir.Var(scope=scope, name=aname, loc=loc)
            block.append(ir.Assign(target=tmp, value=aval, loc=loc))
        # jump to loop entry
        block.append(ir.Jump(target=loopinfo.callfrom, loc=loc))
        return block

    def make_epilogue():
        entry_block = blocks[loopinfo.callfrom]
        scope = entry_block.scope
        loc = entry_block.loc

        block = ir.Block(scope=scope, loc=loc)
        # prepare tuples to return
        vals = [scope.get(name=name) for name in loopinfo.outputs]
        tupexpr = ir.Expr.build_tuple(items=vals, loc=loc)
        tup = scope.make_temp(loc=loc)
        block.append(ir.Assign(target=tup, value=tupexpr, loc=loc))
        # return
        block.append(ir.Return(value=tup, loc=loc))
        return block

    def modify_entry():
        entry_block = blocks[loopinfo.callfrom]
        scope = entry_block.scope
        loc = entry_block.loc

        block = ir.Block(scope=scope, loc=loc)
        # XXX: should fix delete insertion
        # remove deletes at the start
        for start, inst in enumerate(entry_block.body):
            if not isinstance(inst, ir.Del):
                break
        for inst in entry_block.body[start:]:
            block.append(inst)
        return block

    # Lowering assumes the first block to be the one with the smallest offset
    firstblk = min(blocks) - 1
    blocks[firstblk] = make_prologue()
    blocks[loopinfo.callfrom] = modify_entry()
    blocks[loopinfo.returnto] = make_epilogue()


def _loop_lift_modify_blocks(bytecode, loopinfo, blocks,
                             typingctx, targetctx, flags, locals):
    """
    Modify the block inplace to call to the lifted-loop.
    Returns a dictionary of blocks of the lifted-loop.
    """
    from numba.dispatcher import LiftedLoop

    loop = loopinfo.loop
    loopblockkeys = set(loop.body) | set(loop.entries) | set(loop.exits)
    loopblocks = dict((k, blocks[k]) for k in loopblockkeys)

    _loop_lift_prepare_loop_func(loopinfo, loopblocks)

    interp = Interpreter.from_blocks(bytecode=bytecode, blocks=loopblocks,
                                     override_args=loopinfo.inputs,
                                     force_non_generator=True)
    liftedloop = LiftedLoop(interp, typingctx, targetctx, flags, locals)

    # modify for calling into liftedloop
    callblock = _loop_lift_modify_call_block(liftedloop, blocks[loopinfo.callfrom],
                                             loopinfo.inputs, loopinfo.outputs,
                                             loopinfo.returnto)

    # remove blocks
    for k in loopblockkeys:
        del blocks[k]

    blocks[loopinfo.callfrom] = callblock

    return liftedloop


def loop_lifting(interp, typingctx, targetctx, flags, locals):
    """
    Loop lifting transformation.

    Given a interpreter `interp` returns a 2 tuple of
    `(toplevel_interp, [loop0_interp, loop1_interp, ....])`
    """
    blocks = interp.blocks.copy()
    cfg = compute_cfg_from_blocks(blocks)
    loopinfos = _loop_lift_get_candidate_infos(cfg, blocks)
    loops = []
    for loopinfo in loopinfos:
        lifted = _loop_lift_modify_blocks(interp.bytecode, loopinfo, blocks,
                                          typingctx, targetctx, flags, locals)
        loops.append(lifted)
    # make main interpreter
    main = Interpreter.from_blocks(bytecode=interp.bytecode,
                                   blocks=blocks,
                                   used_globals=interp.used_globals)

    return main, loops

