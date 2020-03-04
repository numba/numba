"""
Implement Dominance-Fronter-based SSA by Choi et al
"""
import logging
from copy import copy
from pprint import pformat
from collections import defaultdict

from numba.core import ir, ir_utils
from numba.core.analysis import compute_cfg_from_blocks


_logger = logging.getLogger(__name__)


def recontruct_ssa(fir):
    print("BEFORE SSA".center(80, "-"))
    print(fir.dump())
    print("=" * 80)

    newblocks = _run_ssa(fir.blocks)
    newfir = fir.derive(blocks=newblocks)

    print("AFTER SSA".center(80, "-"))
    print(newfir.dump())
    print("=" * 80)
    return newfir


def _run_ssa(blocks):
    if not blocks:
        # Empty blocks?
        return {}

    violators = _find_defs_violators(blocks)
    for varname, assignlist in violators.items():
        _logger.debug(
            "Fix SSA violator on var %s with %d assignments",
            varname, len(assignlist),
        )

        blocks, defmap = _fresh_vars(blocks, varname, assignlist)
        _logger.debug("Replaced assignments: %s", pformat(defmap))

        blocks = _fix_ssa_vars(blocks, varname, defmap)
    # XXX
    # blocks = _clone_blocks(blocks)
    return blocks


def _fix_ssa_vars(blocks, varname, defmap):
    """Rewrite all uses to ``varname`` given the definition map
    """
    states = _make_states(blocks)
    states['varname'] = varname
    states['defmap'] = defmap
    states['phimap'] = phimap = defaultdict(list)
    states['cfg'] = compute_cfg_from_blocks(blocks)
    newblocks = _run_block_rewrite(blocks, states, _FixSSAVars())
    # insert phi nodes
    for label, philist in phimap.items():
        curblk = newblocks[label]
        # Prepend PHI nodes to the block
        curblk.body = philist + curblk.body
    return newblocks


def _fresh_vars(blocks, varname, assignlist):
    """Rewrite to put fresh variable names
    """
    states = _make_states(blocks)
    states['varname'] = varname
    states['assignlist'] = assignlist
    states['defmap'] = defmap = defaultdict(list)
    newblocks = _run_block_rewrite(blocks, states, _FreshVarHandler())
    return newblocks, defmap


def _get_scope(blocks):
    first, *_ = blocks.values()
    return first.scope


def _find_defs_violators(blocks):
    """
    Returns
    -------
    res : Dict[str, [ir.Assign]]
        The violators in a dictionary of variable name mapping to the
        assignment statements.
    """
    defs = defaultdict(list)
    _run_block_analysis(blocks, defs, _GatherDefsHandler())
    _logger.debug("defs %s", pformat(defs))
    violators = {k: vs for k, vs in defs.items() if len(vs) > 1}
    _logger.debug("SSA violators %s", pformat(violators))
    return violators


def _clone_blocks(blocks):
    states = _make_states(blocks)
    return _run_block_rewrite(blocks, states, _CloneHandler())


def _run_block_analysis(blocks, states, handler):
    for label, blk in blocks.items():
        _logger.debug("==== SSA block analysis pass on %s", label)
        for _ in _run_sbaa_block_pass(states, blk, handler):
            pass


def _run_block_rewrite(blocks, states, handler):
    newblocks = {}
    for label, blk in blocks.items():
        _logger.debug("==== SSA block rewrite pass on %s", label)
        newblk = ir.Block(scope=blk.scope, loc=blk.loc)

        newbody = []
        states['label'] = label
        states['block'] = blk
        for stmt in _run_sbaa_block_pass(states, blk, handler):
            assert stmt is not None
            newbody.append(stmt)
        newblk.body = newbody
        newblocks[label] = newblk
    return newblocks


def _make_states(blocks):
    return dict(
        # cfg=compute_cfg_from_blocks(blocks),
        scope=_get_scope(blocks),
    )


def _run_sbaa_block_pass(states, blk, handler):
    _logger.debug("Running %s", handler)
    for stmt in blk.body:
        _logger.debug("on stmt: %s", stmt)
        if isinstance(stmt, ir.Assign):
            ret = handler.on_assign(states, stmt)
        else:
            ret = handler.on_other(states, stmt)
        if ret is not stmt:
            _logger.debug("replaced with: %s", ret)
        yield ret


class _CloneHandler:
    def on_assign(self, states, assign):
        _logger.debug("    assign to %s", assign.target)
        rhs = assign.value
        if isinstance(rhs, ir.Inst):
            _logger.debug("    used %s", rhs.list_vars())
            # XXX
            replmap = dict(zip(rhs.list_vars(), rhs.list_vars()))
            rhs = ir_utils.replace_vars_inner(rhs, replmap)

        newtarget = assign.target
        return ir.Assign(target=newtarget, value=rhs, loc=assign.loc)

    def on_other(self, states, stmt):
        _logger.debug("    used %s", stmt.list_vars())
        # XXX
        replmap = dict(zip(stmt.list_vars(), stmt.list_vars()))
        stmt = copy(stmt)
        ir_utils.replace_vars_stmt(stmt, replmap)
        return stmt


class _GatherDefsHandler:
    """Find all defs

    ``states`` is a Mapping[str, List[ir.Assign]]
    """
    def on_assign(self, states, assign):
        states[assign.target.name].append(assign)

    def on_other(self, states, stmt):
        pass


class _FreshVarHandler:
    def on_assign(self, states, assign):
        if assign in states['assignlist']:
            scope = states['scope']
            assign = ir.Assign(
                target=scope.redefine(assign.target.name, loc=assign.loc),
                value=assign.value,
                loc=assign.loc
            )
            states['defmap'][states['label']].append(assign)
        return assign

    def on_other(self, states, stmt):
        return stmt


class _FixSSAVars:
    def on_assign(self, states, assign):
        label = states['label']
        defmap = states['defmap']
        defs = defmap[label]
        if assign in defs:
            return assign
        else:
            rhs = assign.value
            if isinstance(rhs, ir.Inst):
                phidef = self._fix_var(
                    states, assign, assign.value.list_vars(),
                )
                if phidef is not None:
                    replmap = {states['varname']: phidef.target}
                    rhs = copy(rhs)
                    ir_utils.replace_vars_inner(rhs, replmap)
                    return ir.Assign(
                        target=assign.target,
                        value=rhs,
                        loc=assign.loc,
                    )
            return assign

    def on_other(self, states, stmt):
        phidef = self._fix_var(
            states, stmt, stmt.list_vars(),
        )
        if phidef is not None:
            replmap = {states['varname']: phidef.target}
            stmt = copy(stmt)
            ir_utils.replace_vars_stmt(stmt, replmap)
        return stmt

    def _fix_var(self, states, stmt, used_vars):
        varnames = [k.name for k in used_vars]
        phivar = states['varname']
        if phivar in varnames:
            return self._find_def(states, stmt)

    def _find_def(self, states, stmt):
        seldef = None
        label = states['label']
        local_defs = states['defmap'][label]
        block = states['block']

        cur_pos = self._stmt_index(stmt, block)
        for defstmt in local_defs:
            def_pos = self._stmt_index(defstmt, block, stop=cur_pos)
            if def_pos < cur_pos:
                seldef = defstmt.target
                break

        if seldef is None:
            seldef = self._find_def_from_top(states, label)

        return seldef

    def _find_def_from_top(self, states, label):
        cfg = states['cfg']
        defmap = states['defmap']
        phimap = states['phimap']
        domfronts = cfg.dominance_frontier()
        for deflabel, defstmt in defmap.items():
            df = domfronts[deflabel]
            if label in df:
                scope = states['scope']
                loc = states['block'].loc
                # fresh variable
                freshvar = scope.redefine(states['varname'],
                                          loc=loc)
                # insert phi
                phinode = ir.Assign(
                    target=freshvar,
                    value=ir.Expr.phi(loc=loc),
                    loc=loc,
                )
                _logger.debug("insert phi node %s at %s", phinode, label)
                defmap[label].insert(0, phinode)
                phimap[label].append(phinode)
                for pred, _ in cfg.predecessors(label):
                    incoming_def = self._find_def_from_bottom(states, pred)
                    phinode.value.incoming_values.append(incoming_def.target)
                    phinode.value.incoming_blocks.append(pred)
                return phinode
        else:
            idom = cfg.immediate_dominators()[label]
            return self._find_def_from_bottom(states, idom)

    def _find_def_from_bottom(self, states, label):
        defmap = states['defmap']
        defs = defmap[label]
        if defs:
            lastdef = defs[-1]
            return lastdef
        else:
            return self._find_def_from_top(states, label)

    def _stmt_index(self, defstmt, block, stop=-1):
        try:
            return block.body.index(defstmt, 0, stop)
        except ValueError:
            return len(block.body)
