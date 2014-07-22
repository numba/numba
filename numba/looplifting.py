from __future__ import print_function, division, absolute_import

from numba import utils
from numba.bytecode import ByteCodeInst, CustomByteCode


def lift_loop(bytecode, dispatcher_factory):
    """Lift the top-level loops.

    Returns (outer, loops)
    ------------------------
    * outer: ByteCode of a copy of the loop-less function.
    * loops: a list of ByteCode of the loops.
    """
    outer = []
    loops = []
    separate_loops(bytecode, outer, loops)

    # Discover variables references
    outer_rds, outer_wrs = find_varnames_uses(bytecode, outer)
    outer_wrs |= set(bytecode.argspec.args)

    dispatchers = []
    outerlabels = set(bytecode.labels)
    outernames = list(bytecode.co_names)

    for loop in loops:
        args, rets = discover_args_and_returns(bytecode, loop, outer_rds,
                                               outer_wrs)
        if rets:
            # Cannot deal with loop that write to variables used in outer body
            # Put the loop back into the outer function
            outer = stitch_instructions(outer, loop)
            # Recompute read-write variable set
            wrs, rds = find_varnames_uses(bytecode, loop)
            outer_wrs |= wrs
            outer_rds |= rds
        else:
            disp = insert_loop_call(bytecode, loop, args,
                                    outer, outerlabels, outernames,
                                    dispatcher_factory)
            dispatchers.append(disp)

    # Build outer bytecode
    codetable = utils.SortedMap((i.offset, i) for i in outer)
    outerbc = CustomByteCode(func=bytecode.func,
                             func_qualname=bytecode.func_qualname,
                             argspec=bytecode.argspec,
                             filename=bytecode.filename,
                             co_names=outernames,
                             co_varnames=bytecode.co_varnames,
                             co_consts=bytecode.co_consts,
                             co_freevars=bytecode.co_freevars,
                             table=codetable,
                             labels=outerlabels & set(codetable.keys()))
    return outerbc, dispatchers


def insert_loop_call(bytecode, loop, args,
                     outer, outerlabels, outernames, dispatcher_factory):
    endloopoffset = loop[-1].next
    # Accepted. Create a bytecode object for the loop
    args = tuple(args)

    lbc = make_loop_bytecode(bytecode, loop, args)

    # Generate dispatcher for this inner loop, and append it to the
    # consts tuple.
    disp = dispatcher_factory(lbc)
    disp_idx = len(bytecode.co_consts)
    bytecode.co_consts += (disp,)

    # Insert jump to the end
    jmp = ByteCodeInst.get(loop[0].offset, 'JUMP_ABSOLUTE',
                           outer[-1].next)
    jmp.lineno = loop[0].lineno
    insert_instruction(outer, jmp)

    outerlabels.add(outer[-1].next)

    # Prepare arguments
    loadfn = ByteCodeInst.get(outer[-1].next, "LOAD_CONST", disp_idx)
    loadfn.lineno = loop[0].lineno
    insert_instruction(outer, loadfn)

    for arg in args:
        loadarg = ByteCodeInst.get(outer[-1].next, 'LOAD_FAST',
                                   bytecode.co_varnames.index(arg))
        loadarg.lineno = loop[0].lineno
        insert_instruction(outer, loadarg)

    # Call function
    assert len(args) < 256
    call = ByteCodeInst.get(outer[-1].next, "CALL_FUNCTION", len(args))
    call.lineno = loop[0].lineno
    insert_instruction(outer, call)

    poptop = ByteCodeInst.get(outer[-1].next, "POP_TOP", None)
    poptop.lineno = loop[0].lineno
    insert_instruction(outer, poptop)

    jmpback = ByteCodeInst.get(outer[-1].next, 'JUMP_ABSOLUTE',
                               endloopoffset)

    jmpback.lineno = loop[0].lineno
    insert_instruction(outer, jmpback)

    return disp


def insert_instruction(insts, item):
    i = find_previous_inst(insts, item.offset)
    insts.insert(i, item)


def find_previous_inst(insts, offset):
    for i, inst in enumerate(insts):
        if inst.offset > offset:
            return i
    return len(insts)


def make_loop_bytecode(bytecode, loop, args):
    # Add return None
    co_consts = tuple(bytecode.co_consts)
    if None not in co_consts:
        co_consts += (None,)

    # Load None
    load_none = ByteCodeInst.get(loop[-1].next, "LOAD_CONST",
                                 co_consts.index(None))
    load_none.lineno = loop[-1].lineno
    loop.append(load_none)

    # Return None
    return_value = ByteCodeInst.get(loop[-1].next, "RETURN_VALUE", 0)
    return_value.lineno = loop[-1].lineno
    loop.append(return_value)

    # Function name
    loop_qualname = bytecode.func_qualname + ".__numba__loop%d__" % loop[0].offset

    # Argspec
    argspectype = type(bytecode.argspec)
    argspec = argspectype(args=args, varargs=(), keywords=(), defaults=())

    # Code table
    codetable = utils.SortedMap((i.offset, i) for i in loop)

    # Custom bytecode object
    lbc = CustomByteCode(func=bytecode.func,
                         func_qualname=loop_qualname,
                         argspec=argspec,
                         filename=bytecode.filename,
                         co_names=bytecode.co_names,
                         co_varnames=bytecode.co_varnames,
                         co_consts=co_consts,
                         co_freevars=bytecode.co_freevars,
                         table=codetable,
                         labels=bytecode.labels)

    return lbc


def stitch_instructions(outer, loop):
    begin = loop[0].offset
    i = find_previous_inst(outer, begin)
    return outer[:i] + loop + outer[i:]


def discover_args_and_returns(bytecode, insts, outer_rds, outer_wrs):
    """
    Basic analysis for args and returns
    This completely ignores the ordering or the read-writes.
    """
    rdnames, wrnames = find_varnames_uses(bytecode, insts)
    # Pass names that are written outside and read locally
    args = outer_wrs & rdnames
    # Return values that it written locally and read outside
    rets = wrnames & outer_rds
    return args, rets


def find_varnames_uses(bytecode, insts):
    rdnames = set()
    wrnames = set()
    for inst in insts:
        if inst.opname == 'LOAD_FAST':
            rdnames.add(bytecode.co_varnames[inst.arg])
        elif inst.opname == 'STORE_FAST':
            wrnames.add(bytecode.co_varnames[inst.arg])
    return rdnames, wrnames


def separate_loops(bytecode, outer, loops):
    """
    Separate top-level loops from the function

    Stores loopless instructions from the original function into `outer`.
    Stores list of loop instructions into `loops`.
    Both `outer` and `loops` are list-like (`append(item)` defined).
    """
    endloop = None
    cur = None
    for inst in bytecode:
        if endloop is None:
            if inst.opname == 'SETUP_LOOP':
                cur = [inst]
                endloop = inst.next + inst.arg
            else:
                outer.append(inst)
        else:
            cur.append(inst)
            if inst.next == endloop:
                for inst in cur:
                    if inst.opname == 'RETURN_VALUE':
                        # Reject if return inside loop
                        outer.extend(cur)
                        break
                else:
                    loops.append(cur)
                endloop = None