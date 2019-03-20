"""
Utils for IR analysis
"""
import operator
from functools import reduce
from collections import namedtuple, defaultdict

from numba import ir
from numba.controlflow import CFGraph
from numba import types, consts

#
# Analysis related to variable lifetime
#

_use_defs_result = namedtuple('use_defs_result', 'usemap,defmap')

# other packages that define new nodes add calls for finding defs
# format: {type:function}
ir_extension_usedefs = {}


def compute_use_defs(blocks):
    """
    Find variable use/def per block.
    """

    var_use_map = {}   # { block offset -> set of vars }
    var_def_map = {}   # { block offset -> set of vars }
    for offset, ir_block in blocks.items():
        var_use_map[offset] = use_set = set()
        var_def_map[offset] = def_set = set()
        for stmt in ir_block.body:
            if type(stmt) in ir_extension_usedefs:
                func = ir_extension_usedefs[type(stmt)]
                func(stmt, use_set, def_set)
                continue
            if isinstance(stmt, ir.Assign):
                if isinstance(stmt.value, ir.Inst):
                    rhs_set = set(var.name for var in stmt.value.list_vars())
                elif isinstance(stmt.value, ir.Var):
                    rhs_set = set([stmt.value.name])
                elif isinstance(stmt.value, (ir.Arg, ir.Const, ir.Global,
                                             ir.FreeVar)):
                    rhs_set = ()
                else:
                    raise AssertionError('unreachable', type(stmt.value))
                # If lhs not in rhs of the assignment
                if stmt.target.name not in rhs_set:
                    def_set.add(stmt.target.name)

            for var in stmt.list_vars():
                # do not include locally defined vars to use-map
                if var.name not in def_set:
                    use_set.add(var.name)

    return _use_defs_result(usemap=var_use_map, defmap=var_def_map)


def compute_live_map(cfg, blocks, var_use_map, var_def_map):
    """
    Find variables that must be alive at the ENTRY of each block.
    We use a simple fix-point algorithm that iterates until the set of
    live variables is unchanged for each block.
    """
    def fix_point_progress(dct):
        """Helper function to determine if a fix-point has been reached.
        """
        return tuple(len(v) for v in dct.values())

    def fix_point(fn, dct):
        """Helper function to run fix-point algorithm.
        """
        old_point = None
        new_point = fix_point_progress(dct)
        while old_point != new_point:
            fn(dct)
            old_point = new_point
            new_point = fix_point_progress(dct)

    def def_reach(dct):
        """Find all variable definition reachable at the entry of a block
        """
        for offset in var_def_map:
            used_or_defined = var_def_map[offset] | var_use_map[offset]
            dct[offset] |= used_or_defined
            # Propagate to outgoing nodes
            for out_blk, _ in cfg.successors(offset):
                dct[out_blk] |= dct[offset]

    def liveness(dct):
        """Find live variables.

        Push var usage backward.
        """
        for offset in dct:
            # Live vars here
            live_vars = dct[offset]
            for inc_blk, _data in cfg.predecessors(offset):
                # Reachable at the predecessor
                reachable = live_vars & def_reach_map[inc_blk]
                # But not defined in the predecessor
                dct[inc_blk] |= reachable - var_def_map[inc_blk]

    live_map = {}
    for offset in blocks.keys():
        live_map[offset] = set(var_use_map[offset])

    def_reach_map = defaultdict(set)
    fix_point(def_reach, def_reach_map)
    fix_point(liveness, live_map)
    return live_map


_dead_maps_result = namedtuple('dead_maps_result', 'internal,escaping,combined')


def compute_dead_maps(cfg, blocks, live_map, var_def_map):
    """
    Compute the end-of-live information for variables.
    `live_map` contains a mapping of block offset to all the living
    variables at the ENTRY of the block.
    """
    # The following three dictionaries will be
    # { block offset -> set of variables to delete }
    # all vars that should be deleted at the start of the successors
    escaping_dead_map = defaultdict(set)
    # all vars that should be deleted within this block
    internal_dead_map = defaultdict(set)
    # all vars that should be deleted after the function exit
    exit_dead_map = defaultdict(set)

    for offset, ir_block in blocks.items():
        # live vars WITHIN the block will include all the locally
        # defined variables
        cur_live_set = live_map[offset] | var_def_map[offset]
        # vars alive in the outgoing blocks
        outgoing_live_map = dict((out_blk, live_map[out_blk])
                                 for out_blk, _data in cfg.successors(offset))
        # vars to keep alive for the terminator
        terminator_liveset = set(v.name
                                 for v in ir_block.terminator.list_vars())
        # vars to keep alive in the successors
        combined_liveset = reduce(operator.or_, outgoing_live_map.values(),
                                  set())
        # include variables used in terminator
        combined_liveset |= terminator_liveset
        # vars that are dead within the block because they are not
        # propagated to any outgoing blocks
        internal_set = cur_live_set - combined_liveset
        internal_dead_map[offset] = internal_set
        # vars that escape this block
        escaping_live_set = cur_live_set - internal_set
        for out_blk, new_live_set in outgoing_live_map.items():
            # successor should delete the unused escaped vars
            new_live_set = new_live_set | var_def_map[out_blk]
            escaping_dead_map[out_blk] |= escaping_live_set - new_live_set

        # if no outgoing blocks
        if not outgoing_live_map:
            # insert var used by terminator
            exit_dead_map[offset] = terminator_liveset

    # Verify that the dead maps cover all live variables
    all_vars = reduce(operator.or_, live_map.values(), set())
    internal_dead_vars = reduce(operator.or_, internal_dead_map.values(),
                                set())
    escaping_dead_vars = reduce(operator.or_, escaping_dead_map.values(),
                                set())
    exit_dead_vars = reduce(operator.or_, exit_dead_map.values(), set())
    dead_vars = (internal_dead_vars | escaping_dead_vars | exit_dead_vars)
    missing_vars = all_vars - dead_vars
    if missing_vars:
        # There are no exit points
        if not cfg.exit_points():
            # We won't be able to verify this
            pass
        else:
            msg = 'liveness info missing for vars: {0}'.format(missing_vars)
            raise RuntimeError(msg)

    combined = dict((k, internal_dead_map[k] | escaping_dead_map[k])
                    for k in blocks)

    return _dead_maps_result(internal=internal_dead_map,
                             escaping=escaping_dead_map,
                             combined=combined)


def compute_live_variables(cfg, blocks, var_def_map, var_dead_map):
    """
    Compute the live variables at the beginning of each block
    and at each yield point.
    The ``var_def_map`` and ``var_dead_map`` indicates the variable defined
    and deleted at each block, respectively.
    """
    # live var at the entry per block
    block_entry_vars = defaultdict(set)

    def fix_point_progress():
        return tuple(map(len, block_entry_vars.values()))

    old_point = None
    new_point = fix_point_progress()

    # Propagate defined variables and still live the successors.
    # (note the entry block automatically gets an empty set)

    # Note: This is finding the actual available variables at the entry
    #       of each block. The algorithm in compute_live_map() is finding
    #       the variable that must be available at the entry of each block.
    #       This is top-down in the dataflow.  The other one is bottom-up.
    while old_point != new_point:
        # We iterate until the result stabilizes.  This is necessary
        # because of loops in the graphself.
        for offset in blocks:
            # vars available + variable defined
            avail = block_entry_vars[offset] | var_def_map[offset]
            # substract variables deleted
            avail -= var_dead_map[offset]
            # add ``avail`` to each successors
            for succ, _data in cfg.successors(offset):
                block_entry_vars[succ] |= avail

        old_point = new_point
        new_point = fix_point_progress()

    return block_entry_vars


#
# Analysis related to controlflow
#

def compute_cfg_from_blocks(blocks):
    cfg = CFGraph()
    for k in blocks:
        cfg.add_node(k)

    for k, b in blocks.items():
        term = b.terminator
        for target in term.get_targets():
            cfg.add_edge(k, target)

    cfg.set_entry_point(min(blocks))
    cfg.process()
    return cfg


def find_top_level_loops(cfg):
    """
    A generator that yields toplevel loops given a control-flow-graph
    """
    blocks_in_loop = set()
    # get loop bodies
    for loop in cfg.loops().values():
        insiders = set(loop.body) | set(loop.entries) | set(loop.exits)
        insiders.discard(loop.header)
        blocks_in_loop |= insiders
    # find loop that is not part of other loops
    for loop in cfg.loops().values():
        if loop.header not in blocks_in_loop:
            yield loop


# Functions to manipulate IR
def dead_branch_prune(func_ir, called_args):
    """
    Removes dead branches based on constant inference from function args.
    This directly mutates the IR.

    func_ir is the IR
    called_args are the actual arguments with which the function is called
    """
    from .ir_utils import get_definition, guard, find_const, GuardException

    DEBUG = 0

    def find_branches(func_ir):
        # find *all* branches
        branches = []
        for blk in func_ir.blocks.values():
            branch_or_jump = blk.body[-1]
            if isinstance(branch_or_jump, ir.Branch):
                branch = branch_or_jump
                condition = guard(get_definition, func_ir, branch.cond.name)
                if condition is not None:
                    branches.append((branch, condition, blk))
        return branches

    def do_prune(take_truebr, blk):
        keep = branch.truebr if take_truebr else branch.falsebr
        # replace the branch with a direct jump
        jmp = ir.Jump(keep, loc=branch.loc)
        blk.body[-1] = jmp
        return 1 if keep == branch.truebr else 0

    def prune_by_type(branch, condition, blk, *conds):
        # this prunes a given branch and fixes up the IR
        # at least one needs to be a NoneType
        lhs_cond, rhs_cond = conds
        lhs_none = isinstance(lhs_cond, types.NoneType)
        rhs_none = isinstance(rhs_cond, types.NoneType)
        if lhs_none or rhs_none:
            take_truebr = condition.fn(lhs_cond, rhs_cond)
            if DEBUG > 0:
                kill = branch.falsebr if take_truebr else branch.truebr
                print("Pruning %s" % kill, branch, lhs_cond, rhs_cond,
                      condition.fn)
            taken = do_prune(take_truebr, blk)
            return True, taken
        return False, None

    def prune_by_value(branch, condition, blk, *conds):
        lhs_cond, rhs_cond = conds
        take_truebr = condition.fn(lhs_cond, rhs_cond)
        if DEBUG > 0:
            kill = branch.falsebr if take_truebr else branch.truebr
            print("Pruning %s" % kill, branch, lhs_cond, rhs_cond, condition.fn)
        taken = do_prune(take_truebr, blk)
        return True, taken

    class Unknown(object):
        pass

    def resolve_input_arg_const(input_arg):
        """
        Resolves an input arg to a constant (if possible)
        """
        idx = func_ir.arg_names.index(input_arg)
        input_arg_ty = called_args[idx]

        # comparing to None?
        if isinstance(input_arg_ty, types.NoneType):
            return input_arg_ty

        # is it a kwarg default
        if isinstance(input_arg_ty, types.Omitted):
            val = input_arg_ty.value
            if isinstance(val, types.NoneType):
                return val
            elif val is None:
                return types.NoneType('none')

        # literal type, return the type itself so comparisons like `x == None`
        # still work as e.g. x = types.int64 will never be None/NoneType so
        # the branch can still be pruned
        return getattr(input_arg_ty, 'literal_type', Unknown())

    if DEBUG > 1:
        print("before".center(80, '-'))
        print(func_ir.dump())

    # This looks for branches where:
    # at least one arg of the condition is in input args and const
    # at least one an arg of the condition is a const
    # if the condition is met it will replace the branch with a jump
    branch_info = find_branches(func_ir)
    nullified_conditions = [] # stores conditions that have no impact post prune
    for branch, condition, blk in branch_info:
        const_conds = []
        if isinstance(condition, ir.Expr) and condition.op == 'binop':
            prune = prune_by_value
            for arg in [condition.lhs, condition.rhs]:
                resolved_const = Unknown()
                if arg.name in func_ir.arg_names:
                    # it's an e.g. literal argument to the function
                    resolved_const = resolve_input_arg_const(arg.name)
                    prune = prune_by_type
                else:
                    # it's some const argument to the function, cannot use guard
                    # here as the const itself may be None
                    try:
                        resolved_const = find_const(func_ir, arg)
                        if resolved_const is None:
                            resolved_const = types.NoneType('none')
                    except GuardException:
                        pass

                if not isinstance(resolved_const, Unknown):
                    const_conds.append(resolved_const)

            # lhs/rhs are consts
            if len(const_conds) == 2:
                # prune the branch, switch the branch for an unconditional jump
                prune_stat, taken = prune(branch, condition, blk, *const_conds)
                if(prune_stat):
                    # add the condition to the list of nullified conditions
                    nullified_conditions.append((condition, taken))

    # 'ERE BE DRAGONS...
    # It is the evaluation of the condition expression that often trips up type
    # inference, so ideally it would be removed as it is effectively rendered
    # dead by the unconditional jump if a branch was pruned. However, there may
    # be references to the condition that exist in multiple places (e.g. dels)
    # and we cannot run DCE here as typing has not taken place to give enough
    # information to run DCE safely. Upshot of all this is the condition gets
    # rewritten below into a benign const that typing will be happy with and DCE
    # can remove it and its reference post typing when it is safe to do so
    # (if desired). It is required that the const is assigned a value that
    # indicates the branch taken as its mutated value would be read in the case
    # of object mode fall back in place of the condition itself. For
    # completeness the func_ir._definitions and ._consts are also updated to
    # make the IR state self consistent.

    deadcond = [x[0] for x in nullified_conditions]
    for _, cond, blk in branch_info:
        if cond in deadcond:
            for x in blk.body:
                if isinstance(x, ir.Assign) and x.value is cond:
                    # rewrite the condition as a true/false bit
                    branch_bit = nullified_conditions[deadcond.index(cond)][1]
                    x.value = ir.Const(branch_bit, loc=x.loc)
                    # update the specific definition to the new const
                    defns = func_ir._definitions[x.target.name]
                    repl_idx = defns.index(cond)
                    defns[repl_idx] = x.value

    # Remove dead blocks, this is safe as it relies on the CFG only.
    cfg = compute_cfg_from_blocks(func_ir.blocks)
    for dead in cfg.dead_nodes():
        del func_ir.blocks[dead]

    # if conditions were nullified then consts were rewritten, update
    if nullified_conditions:
        func_ir._consts = consts.ConstantInference(func_ir)

    if DEBUG > 1:
        print("after".center(80, '-'))
        print(func_ir.dump())
