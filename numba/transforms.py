"""
Implement transformation on Numba IR
"""

from __future__ import absolute_import, print_function

from pprint import pprint
from functools import singledispatch
from collections import namedtuple

from numba.controlflow import CFGraph
from numba.analysis import compute_use_defs, compute_live_map
from numba import ir


def _build_controlflow(blocks):
    cfg = CFGraph()
    for k in blocks:
        cfg.add_node(k)

    for k, b in blocks.items():
        term = b.terminator
        for target in _get_targets(term):
            cfg.add_edge(k, target)

    cfg.set_entry_point(min(blocks))
    cfg.process()
    return cfg


@singledispatch
def _get_targets(term):
    raise NotImplementedError(type(term))


@_get_targets.register(ir.Jump)
def _(term):
    return [term.target]


@_get_targets.register(ir.Return)
def _(term):
    return []


@_get_targets.register(ir.Branch)
def _(term):
    return [term.truebr, term.falsebr]


def _extract_loop_lifting_candidates(cfg):
    """
    Returns a list of loops that are candidate for loop lifting
    """
    toplevelloops = []
    blocks_in_loop = set()
    # get loop bodies
    for loop in cfg.loops().values():
        insiders = set(loop.body) | set(loop.entries) | set(loop.exits)
        insiders.discard(loop.header)
        blocks_in_loop |= insiders
    # find loop that is not part of other loops
    for loop in cfg.loops().values():
        if loop.header not in blocks_in_loop:
            toplevelloops.append(loop)

    # XXX: break this function into two
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

    return [loop for loop in toplevelloops
            if same_exit_point(loop) and one_entry(loop)]


_loop_lift_info = namedtuple('loop_lift_info',
                             'loop,inputs,outputs,callfrom,returnto')


def _loop_lift_get_infos_for_lifted_loops(blocks):
    cfg = _build_controlflow(blocks)
    loops = _extract_loop_lifting_candidates(cfg)

    usedefs = compute_use_defs(blocks)
    livemap = compute_live_map(cfg, blocks, usedefs.usemap, usedefs.defmap)

    loopinfos = []
    for loop in loops:
        [callfrom] = loop.entries   # requirement checked earlier
        an_exit = next(iter(loop.exits))
        [returnto] = cfg.descendents(an_exit)  # requirement checked earlier

        # note: sorted for stable ordering
        inputs = sorted(livemap[callfrom])
        outputs = sorted(livemap[returnto])
        loopinfos.append(_loop_lift_info(loop=loop,
                                         inputs=inputs, outputs=outputs,
                                         callfrom=callfrom, returnto=returnto))

    return loopinfos


def _loop_lift_modify_call_block(liftedloop, block, inputs, outputs, returnto):
    scope = block.scope
    loc = block.loc
    blk = ir.Block(scope=scope, loc=loc)
    # load loop
    fn = ir.Global(name='liftedloop.%s' % id(liftedloop), value=liftedloop,
                   loc=loc)
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
    blk.append(ir.Del(value=fnvar, loc=loc))
    blk.append(ir.Del(value=callres, loc=loc))

    # jump to next block
    blk.append(ir.Jump(target=returnto, loc=loc))
    return blk


def _loop_lift_prepare_loop_func(loopinfo, blocks):

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

    # Lowering assumes the first block to be the one with the smallest offset
    firstblk = min(blocks) - 1
    blocks[firstblk] = make_prologue()
    blocks[loopinfo.returnto] = make_epilogue()


def _loop_lift_modify_blocks(bytecode, loopinfo, blocks):
    """
    Modify the block inplace to call to the lifted-loop.
    Returns a dictionary of blocks of the lifted-loop.
    """
    loop = loopinfo.loop
    loopblockkeys = set(loop.body) | set(loop.entries) | set(loop.exits)
    loopblocks = dict((k, blocks[k]) for k in loopblockkeys)

    _loop_lift_prepare_loop_func(loopinfo, loopblocks)
    liftedloop = LiftedLoop(bytecode, loopblocks, loopinfo.inputs)

    # modify for calling into liftedloop
    callblock = _loop_lift_modify_call_block(liftedloop, blocks[loopinfo.callfrom],
                                             loopinfo.inputs, loopinfo.outputs,
                                             loopinfo.returnto)

    # remove blocks
    for k in loopblockkeys:
        del blocks[k]

    blocks[loopinfo.callfrom] = callblock

    return liftedloop


def loop_lifting(interp):
    from numba.interpreter import Interpreter

    blocks = interp.blocks
    loopinfos = _loop_lift_get_infos_for_lifted_loops(blocks)
    loops = []
    for loopinfo in loopinfos:
        loops.append(_loop_lift_modify_blocks(interp.bytecode, loopinfo, blocks))
    # make main interpreter
    # XXX duplication
    main = Interpreter(bytecode=interp.bytecode)
    main.loc = interp.loc
    main.arg_count = interp.arg_count
    main.arg_names = interp.arg_names
    main.blocks = blocks

    return main, loops


class LiftedLoop(object):
    # TODO
    def __init__(self, bytecode, blocks, argnames):
        from numba.interpreter import Interpreter

        self.interp = Interpreter(bytecode=bytecode)
        firstblock = blocks[min(blocks)]
        self.interp.blocks = blocks
        self.interp.loc = firstblock.loc
        self.interp.arg_names = argnames
        self.interp.arg_count = len(argnames)

