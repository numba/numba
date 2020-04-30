"""
Implement Dominance-Fronter-based SSA by Choi et al described in Inria SSA book

References:

- Static Single Assignment Book by Inria
  http://ssabook.gforge.inria.fr/latest/book.pdf
- Choi et al. Incremental computation of static single assignment form.
"""
import logging
import operator
import warnings
from functools import reduce
from copy import copy
from pprint import pformat
from collections import defaultdict

from numba import config
from numba.core import ir, ir_utils, errors
from numba.core.analysis import compute_cfg_from_blocks


_logger = logging.getLogger(__name__)


def reconstruct_ssa(func_ir):
    """Apply SSA reconstruction algorithm on the given IR.

    Produces minimal SSA using Choi et al algorithm.
    """
    _logger.debug("BEFORE SSA".center(80, "-"))
    _logger.debug(func_ir.dump_to_string())
    _logger.debug("=" * 80)

    func_ir.blocks = _run_ssa(func_ir.blocks)

    _logger.debug("AFTER SSA".center(80, "-"))
    _logger.debug(func_ir.dump_to_string())
    _logger.debug("=" * 80)
    return func_ir


def _run_ssa(blocks):
    """Run SSA reconstruction on IR blocks of a function.
    """
    if not blocks:
        # Empty blocks?
        return {}

    # Find SSA violators
    violators = _find_defs_violators(blocks)
    # Process one SSA-violating variable at a time
    for varname in violators:
        _logger.debug(
            "Fix SSA violator on var %s", varname,
        )
        # Fix up the LHS
        # Put fresh variables for all assignments to the variable
        blocks, defmap = _fresh_vars(blocks, varname)
        _logger.debug("Replaced assignments: %s", pformat(defmap))
        # Fix up the RHS
        # Re-associate the variable uses with the reaching definition
        blocks = _fix_ssa_vars(blocks, varname, defmap)
    return blocks


def _fix_ssa_vars(blocks, varname, defmap):
    """Rewrite all uses to ``varname`` given the definition map
    """
    states = _make_states(blocks)
    states['varname'] = varname
    states['defmap'] = defmap
    states['phimap'] = phimap = defaultdict(list)
    states['cfg'] = cfg = compute_cfg_from_blocks(blocks)
    states['df+'] = _iterated_domfronts(cfg)
    newblocks = _run_block_rewrite(blocks, states, _FixSSAVars())
    # check for unneeded phi nodes
    _remove_unneeded_phis(phimap)
    # insert phi nodes
    for label, philist in phimap.items():
        curblk = newblocks[label]
        # Prepend PHI nodes to the block
        curblk.body = philist + curblk.body
    return newblocks


def _remove_unneeded_phis(phimap):
    """Remove unneeded PHIs from the phimap
    """
    all_phis = []
    for philist in phimap.values():
        all_phis.extend(philist)
    unneeded_phis = set()
    # Find unneeded PHIs.
    for phi in all_phis:
        ivs = phi.value.incoming_values
        # It's unneeded if the incomings are either undefined or
        # the PHI node target is itself
        if all(iv is ir.UNDEFINED or iv == phi.target for iv in ivs):
            unneeded_phis.add(phi)
    # Fix up references to unneeded PHIs
    for phi in all_phis:
        for unneed in unneeded_phis:
            if unneed is not phi:
                # If the unneeded PHI is in the current phi's incoming values
                if unneed.target in phi.value.incoming_values:
                    # Replace the unneeded PHI with an UNDEFINED
                    idx = phi.value.incoming_values.index(unneed.target)
                    phi.value.incoming_values[idx] = ir.UNDEFINED
    # Remove unneeded phis
    for philist in phimap.values():
        for unneeded in unneeded_phis:
            if unneeded in philist:
                philist.remove(unneeded)


def _iterated_domfronts(cfg):
    """Compute the iterated dominance frontiers (DF+ in literatures).

    Returns a dictionary which maps block label to the set of labels of its
    iterated dominance frontiers.
    """
    domfronts = {k: set(vs) for k, vs in cfg.dominance_frontier().items()}
    keep_going = True
    while keep_going:
        keep_going = False
        for k, vs in domfronts.items():
            inner = reduce(operator.or_, [domfronts[v] for v in vs], set())
            if inner.difference(vs):
                vs |= inner
                keep_going = True
    return domfronts


def _fresh_vars(blocks, varname):
    """Rewrite to put fresh variable names
    """
    states = _make_states(blocks)
    states['varname'] = varname
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
    res : Set[str]
        The SSA violators in a dictionary of variable names.
    """
    defs = defaultdict(list)
    _run_block_analysis(blocks, defs, _GatherDefsHandler())
    _logger.debug("defs %s", pformat(defs))
    violators = {k for k, vs in defs.items() if len(vs) > 1}
    _logger.debug("SSA violators %s", pformat(violators))
    return violators


def _run_block_analysis(blocks, states, handler):
    for label, blk in blocks.items():
        _logger.debug("==== SSA block analysis pass on %s", label)
        for _ in _run_ssa_block_pass(states, blk, handler):
            pass


def _run_block_rewrite(blocks, states, handler):
    newblocks = {}
    for label, blk in blocks.items():
        _logger.debug("==== SSA block rewrite pass on %s", label)
        newblk = ir.Block(scope=blk.scope, loc=blk.loc)

        newbody = []
        states['label'] = label
        states['block'] = blk
        for stmt in _run_ssa_block_pass(states, blk, handler):
            assert stmt is not None
            newbody.append(stmt)
        newblk.body = newbody
        newblocks[label] = newblk
    return newblocks


def _make_states(blocks):
    return dict(
        scope=_get_scope(blocks),
    )


def _run_ssa_block_pass(states, blk, handler):
    _logger.debug("Running %s", handler)
    for stmt in blk.body:
        _logger.debug("on stmt: %s", stmt)
        if isinstance(stmt, ir.Assign):
            ret = handler.on_assign(states, stmt)
        else:
            ret = handler.on_other(states, stmt)
        if ret is not stmt and ret is not None:
            _logger.debug("replaced with: %s", ret)
        yield ret


class _BaseHandler:
    """A base handler for all the passes used here for the SSA algorithm.
    """
    def on_assign(self, states, assign):
        """
        Called when the pass sees an ``ir.Assign``.

        Subclasses should override this for custom behavior

        Parameters
        -----------
        states : dict
        assign : numba.ir.Assign

        Returns
        -------
        stmt : numba.ir.Assign or None
            For rewrite passes, the return value is used as the replacement
            for the given statement.
        """

    def on_other(self, states, stmt):
        """
        Called when the pass sees an ``ir.Stmt`` that's not an assignment.

        Subclasses should override this for custom behavior

        Parameters
        -----------
        states : dict
        assign : numba.ir.Stmt

        Returns
        -------
        stmt : numba.ir.Stmt or None
            For rewrite passes, the return value is used as the replacement
            for the given statement.
        """


class _GatherDefsHandler(_BaseHandler):
    """Find all defs

    ``states`` is a Mapping[str, List[ir.Assign]]
    """
    def on_assign(self, states, assign):
        states[assign.target.name].append(assign)


class UndefinedVariable:
    def __init__(self):
        raise NotImplementedError("Not intended for instantiation")

    target = ir.UNDEFINED


class _FreshVarHandler(_BaseHandler):
    """Replaces assignment target with new fresh variables.
    """
    def on_assign(self, states, assign):
        if assign.target.name == states['varname']:
            scope = states['scope']
            defmap = states['defmap']
            # Allow first assignment to retain the name
            if len(defmap) == 0:
                newtarget = assign.target
                _logger.debug("first assign: %s", newtarget)
            else:
                newtarget = scope.redefine(assign.target.name, loc=assign.loc)
            assign = ir.Assign(
                target=newtarget,
                value=assign.value,
                loc=assign.loc
            )
            defmap[states['label']].append(assign)
        return assign

    def on_other(self, states, stmt):
        return stmt


class _FixSSAVars(_BaseHandler):
    """Replace variable uses in IR nodes to the correct reaching variable
    and introduce Phi nodes if necessary. This class contains the core of
    the SSA reconstruction algorithm.

    See Ch 5 of the Inria SSA book for reference. The method names used here
    are similar to the names used in the pseudocode in the book.
    """
    def on_assign(self, states, assign):
        rhs = assign.value
        if isinstance(rhs, ir.Inst):
            newdef = self._fix_var(
                states, assign, assign.value.list_vars(),
            )
            # Has a replacement that is not the current variable
            if newdef is not None and newdef.target is not ir.UNDEFINED:
                if states['varname'] != newdef.target.name:
                    replmap = {states['varname']: newdef.target}
                    rhs = copy(rhs)

                    ir_utils.replace_vars_inner(rhs, replmap)
                    return ir.Assign(
                        target=assign.target,
                        value=rhs,
                        loc=assign.loc,
                    )
        elif isinstance(rhs, ir.Var):
            newdef = self._fix_var(states, assign, [rhs])
            # Has a replacement that is not the current variable
            if newdef is not None and newdef.target is not ir.UNDEFINED:
                if states['varname'] != newdef.target.name:
                    return ir.Assign(
                        target=assign.target,
                        value=newdef.target,
                        loc=assign.loc,
                    )

        return assign

    def on_other(self, states, stmt):
        newdef = self._fix_var(
            states, stmt, stmt.list_vars(),
        )
        if newdef is not None and newdef.target is not ir.UNDEFINED:
            if states['varname'] != newdef.target.name:
                replmap = {states['varname']: newdef.target}
                stmt = copy(stmt)
                ir_utils.replace_vars_stmt(stmt, replmap)
        return stmt

    def _fix_var(self, states, stmt, used_vars):
        """Fix all variable uses in ``used_vars``.
        """
        varnames = [k.name for k in used_vars]
        phivar = states['varname']
        if phivar in varnames:
            return self._find_def(states, stmt)

    def _find_def(self, states, stmt):
        """Find definition of ``stmt`` for the statement ``stmt``
        """
        _logger.debug("find_def var=%r stmt=%s", states['varname'], stmt)
        selected_def = None
        label = states['label']
        local_defs = states['defmap'][label]
        local_phis = states['phimap'][label]
        block = states['block']

        cur_pos = self._stmt_index(stmt, block)
        for defstmt in reversed(local_defs):
            # Phi nodes have no index
            def_pos = self._stmt_index(defstmt, block, stop=cur_pos)
            if def_pos < cur_pos:
                selected_def = defstmt
                break
            # Maybe it's a PHI
            elif defstmt in local_phis:
                selected_def = local_phis[-1]
                break

        if selected_def is None:
            selected_def = self._find_def_from_top(
                states, label, loc=stmt.loc,
            )
        return selected_def

    def _find_def_from_top(self, states, label, loc):
        """Find definition reaching block of ``label``.

        This method would look at all dominance frontiers.
        Insert phi node if necessary.
        """
        _logger.debug("find_def_from_top label %r", label)
        cfg = states['cfg']
        defmap = states['defmap']
        phimap = states['phimap']
        domfronts = states['df+']
        for deflabel, defstmt in defmap.items():
            df = domfronts[deflabel]
            if label in df:
                scope = states['scope']
                loc = states['block'].loc
                # fresh variable
                freshvar = scope.redefine(states['varname'], loc=loc)
                # insert phi
                phinode = ir.Assign(
                    target=freshvar,
                    value=ir.Expr.phi(loc=loc),
                    loc=loc,
                )
                _logger.debug("insert phi node %s at %s", phinode, label)
                defmap[label].insert(0, phinode)
                phimap[label].append(phinode)
                # Find incoming values for the Phi node
                for pred, _ in cfg.predecessors(label):
                    incoming_def = self._find_def_from_bottom(
                        states, pred, loc=loc,
                    )
                    _logger.debug("incoming_def %s", incoming_def)
                    phinode.value.incoming_values.append(incoming_def.target)
                    phinode.value.incoming_blocks.append(pred)
                return phinode
        else:
            idom = cfg.immediate_dominators()[label]
            if idom == label:
                # We have searched to the top of the idom tree.
                # Since we still cannot find a definition,
                # we will warn.
                _warn_about_uninitialized_variable(states['varname'], loc)
                return UndefinedVariable
            _logger.debug("idom %s from label %s", idom, label)
            return self._find_def_from_bottom(states, idom, loc=loc)

    def _find_def_from_bottom(self, states, label, loc):
        """Find definition from within the block at ``label``.
        """
        _logger.debug("find_def_from_bottom label %r", label)
        defmap = states['defmap']
        defs = defmap[label]
        if defs:
            lastdef = defs[-1]
            return lastdef
        else:
            return self._find_def_from_top(states, label, loc=loc)

    def _stmt_index(self, defstmt, block, stop=-1):
        """Find the positional index of the statement at ``block``.

        Assumptions:
        - no two statements can point to the same object.
        """
        # Compare using id() as IR node equality is for semantic equivalence
        # opposed to direct equality (the location and scope are not considered
        # as part of the equality measure, this is important here).
        for i in range(len(block.body))[:stop]:
            if block.body[i] is defstmt:
                return i
        return len(block.body)


def _warn_about_uninitialized_variable(varname, loc):
    if config.ALWAYS_WARN_UNINIT_VAR:
        warnings.warn(
            errors.NumbaWarning(
                f"Detected uninitialized variable {varname}",
                loc=loc),
        )
