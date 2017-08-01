"""
Utils for IR analysis
"""
import operator
from functools import reduce
from collections import namedtuple, defaultdict

from numba import ir
from numba.controlflow import CFGraph

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
    live_map = {}
    for offset in blocks.keys():
        live_map[offset] = var_use_map[offset]

    def fix_point_progress():
        return tuple(len(v) for v in live_map.values())

    old_point = None
    new_point = fix_point_progress()
    while old_point != new_point:
        for offset in live_map.keys():
            for inc_blk, _data in cfg.predecessors(offset):
                # substract all variables that are defined in
                # the incoming block
                live_map[inc_blk] |= live_map[offset] - var_def_map[inc_blk]
        old_point = new_point
        new_point = fix_point_progress()

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
    # all vars that should be delted after the function exit
    exit_dead_map = defaultdict(set)

    for offset, ir_block in blocks.items():
        # live vars WITHIN the block will include all the locally
        # defined variables
        cur_live_set = live_map[offset] | var_def_map[offset]
        # vars alive alive in the outgoing blocks
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
        # vars that are dead within the block beacuse they are not
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
