from __future__ import print_function, division, absolute_import

from numba import utils
from numba.bytecode import ByteCodeInst, CustomByteCode
from collections import defaultdict


def lift_loop(bytecode, dispatcher_factory):
    """Lift the top-level loops.

    Returns (outer, loops)
    ------------------------
    * outer: ByteCode of a copy of the loop-less function.
    * loops: a list of ByteCode of the loops.
    """
    outer = []
    loops = []
    # Discover variables references
    outer_rds, outer_wrs = find_varnames_uses(bytecode, iter(bytecode))
    # Separate loops and outer
    separate_loops(bytecode, outer, loops)

    # Prepend arguments as negative bytecode offset
    for a in bytecode.pysig.parameters:
        outer_wrs[a] = [-1] + outer_wrs[a]

    dispatchers = []
    outerlabels = set(bytecode.labels)
    outernames = list(bytecode.co_names)

    for loop in loops:
        args, rets = discover_args_and_returns(bytecode, loop, outer_rds,
                                               outer_wrs)

        disp = insert_loop_call(bytecode, loop, args,
                                outer, outerlabels, rets,
                                dispatcher_factory)
        dispatchers.append(disp)

    # Build outer bytecode
    codetable = utils.SortedMap((i.offset, i) for i in outer)
    outerbc = CustomByteCode(func=bytecode.func,
                             func_qualname=bytecode.func_qualname,
                             is_generator=bytecode.is_generator,
                             pysig=bytecode.pysig,
                             filename=bytecode.filename,
                             co_names=outernames,
                             co_varnames=bytecode.co_varnames,
                             co_consts=bytecode.co_consts,
                             co_freevars=bytecode.co_freevars,
                             table=codetable,
                             labels=outerlabels & set(codetable.keys()))

    return outerbc, dispatchers

@utils.total_ordering
class SubOffset(object):
    """The loop-jitting may insert bytecode between two bytecode but we
    cannot guarantee that there is enough integral space between two offsets.
    This class workaround the problem by introducing a fractional part to the
    offset.
    """
    def __init__(self, val, sub=1):
        assert sub > 0, "fractional part cannot be <= 0"
        self.val = val
        self.sub = sub

    def next(self):
        """Helper method to get the next suboffset by incrementing the
        fractional part only
        """
        return SubOffset(self.val, self.sub + 1)

    def __add__(self, other):
        """Adding to a suboffset will only increment the fractional part.
        The integral part is immutable.
        """
        return SubOffset(self.val, self.sub + other)

    def __hash__(self):
        return hash((self.val, self.sub))

    def __lt__(self, other):
        """Can only compare to SubOffset or int
        """
        if isinstance(other, SubOffset):
            if self.val < other.val:
                return self
            elif self.val == other.val:
                return self.sub < other.sub
            else:
                return False
        elif isinstance(other, int):
            return self.val < other
        else:
            return NotImplemented

    def __eq__(self, other):
        if isinstance(other, SubOffset):
            return self.val == other.val and self.sub == other.sub
        elif isinstance(other, int):
            # Can never be equal to a integer by definition
            return False
        else:
            return NotImplemented

    def __repr__(self):
        """Print like a floating-point by it is not one at all.
        """
        return "{0}.{1}".format(self.val, self.sub)


def insert_loop_call(bytecode, loop, args, outer, outerlabels, returns,
                     dispatcher_factory):
    endloopoffset = loop[-1].next
    # Accepted. Create a bytecode object for the loop
    args = tuple(args)

    lbc = make_loop_bytecode(bytecode, loop, args, returns)

    # Generate dispatcher for this inner loop, and append it to the
    # consts tuple.
    disp = dispatcher_factory(lbc)
    disp_idx = len(bytecode.co_consts)
    bytecode.co_consts += (disp,)

    # Insert jump to the end
    insertpt = SubOffset(loop[0].next)
    jmp = ByteCodeInst.get(loop[0].offset, 'JUMP_ABSOLUTE', insertpt)
    jmp.lineno = loop[0].lineno
    insert_instruction(outer, jmp)

    outerlabels.add(outer[-1].next)

    # Prepare arguments
    loadfn = ByteCodeInst.get(insertpt, "LOAD_CONST", disp_idx)
    loadfn.lineno = loop[0].lineno
    insert_instruction(outer, loadfn)

    insertpt = insertpt.next()
    for arg in args:
        loadarg = ByteCodeInst.get(insertpt, 'LOAD_FAST',
                                   bytecode.co_varnames.index(arg))
        loadarg.lineno = loop[0].lineno
        insert_instruction(outer, loadarg)
        insertpt = insertpt.next()

    # Call function
    assert len(args) < 256
    call = ByteCodeInst.get(insertpt, "CALL_FUNCTION", len(args))
    call.lineno = loop[0].lineno
    insert_instruction(outer, call)

    insertpt = insertpt.next()

    if returns:
        # Unpack arguments
        unpackseq = ByteCodeInst.get(insertpt, "UNPACK_SEQUENCE",
                                  len(returns))
        unpackseq.lineno = loop[0].lineno
        insert_instruction(outer, unpackseq)
        insertpt = insertpt.next()

        for out in returns:
            # Store each variable
            storefast = ByteCodeInst.get(insertpt, "STORE_FAST",
                                      bytecode.co_varnames.index(out))
            storefast.lineno = loop[0].lineno
            insert_instruction(outer, storefast)
            insertpt = insertpt.next()
    else:
        # No return value
        poptop = ByteCodeInst.get(insertpt, "POP_TOP", None)
        poptop.lineno = loop[0].lineno
        insert_instruction(outer, poptop)
        insertpt = insertpt.next()

    jmpback = ByteCodeInst.get(insertpt, 'JUMP_ABSOLUTE',
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


def make_loop_bytecode(bytecode, loop, args, returns):
    # Add return None
    co_consts = tuple(bytecode.co_consts)
    if None not in co_consts:
        co_consts += (None,)

    if returns:
        for out in returns:
            # Load output
            loadfast = ByteCodeInst.get(loop[-1].next, "LOAD_FAST",
                                         bytecode.co_varnames.index(out))
            loadfast.lineno = loop[-1].lineno
            loop.append(loadfast)

        # Build tuple
        buildtuple = ByteCodeInst.get(loop[-1].next, "BUILD_TUPLE",
                                    len(returns))
        buildtuple.lineno = loop[-1].lineno
        loop.append(buildtuple)

    else:
        # Load None
        load_none = ByteCodeInst.get(loop[-1].next, "LOAD_CONST",
                                     co_consts.index(None))
        load_none.lineno = loop[-1].lineno
        loop.append(load_none)

    # Return TOS
    return_value = ByteCodeInst.get(loop[-1].next, "RETURN_VALUE", 0)
    return_value.lineno = loop[-1].lineno
    loop.append(return_value)

    # Function name
    loop_qualname = bytecode.func_qualname + ".__numba__loop%d__" % loop[0].offset

    # Code table
    codetable = utils.SortedMap((i.offset, i) for i in loop)

    # Custom bytecode object
    lbc = CustomByteCode(func=bytecode.func,
                         func_qualname=loop_qualname,
                         # Enforced in separate_loops()
                         is_generator=False,
                         pysig=bytecode.pysig,
                         arg_count=len(args),
                         arg_names=args,
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


def remove_from_outer_use(inneruse, outeruse):
    for name in inneruse:
        inuse = inneruse[name]
        outuse = outeruse[name]
        outeruse[name] = sorted(list(set(outuse) - set(inuse)))


def discover_args_and_returns(bytecode, insts, outer_rds, outer_wrs):
    """
    Basic analysis for args and returns
    This completely ignores the ordering or the read-writes.

    outer_rds and outer_wrs are modified

    Note:
    An invalid argument and return set will likely to cause a RuntimeWarning
    in the dataflow analysis due to mismatch in stack offset.
    """
    rdnames, wrnames = find_varnames_uses(bytecode, insts)

    # Remove all local use from the set
    remove_from_outer_use(rdnames, outer_rds)
    remove_from_outer_use(wrnames, outer_wrs)

    # Return every variables that are written inside the loop and read
    # afterwards
    rets = set()
    for name, uselist in wrnames.items():
        if name in outer_rds:
            endofloop = insts[-1].offset

            # Find the next read
            for nextrd in outer_rds[name]:
                if nextrd > endofloop:
                    break
            else:
                nextrd = None

            # Find the next write
            for nextwr in outer_wrs[name]:
                if nextwr > endofloop:
                    break
            else:
                nextwr = None

            # If there is a read but no write OR
            # If the next use is a read, THEN
            # it is a return value
            if nextrd is not None and (nextwr is None or nextwr > nextrd):
                rets.add(name)

    # Make variables arguments if they are read before defined before the loop.
    # Since we can't tell if things are conditionally defined here,
    # We will have to be more conservative.
    args = set()
    firstline = insts[0].offset
    for name in rdnames.keys():
        outer_write = outer_wrs[name]
        # If there exists a definition before the start of the loop
        # for a variable read in side the loop.
        if any(i < firstline for i in outer_write):
            args.add(name)

    # Make variables arguments if it is being returned but defined before the
    # loop.
    for name in rets:
        if any(i < insts[0].offset for i in outer_wrs[name]):
            args.add(name)

    # Re-add the arguments back to outer_rds
    for name in args:
        outer_rds[name] = sorted(set(outer_rds[name]) | set([firstline]))

    # Re-add the arguments back to outer_wrs
    for name in rets:
        outer_wrs[name] = sorted(set(outer_wrs[name]) | set([firstline]))

    return args, rets


def find_varnames_uses(bytecode, insts):
    rdnames = defaultdict(list)
    wrnames = defaultdict(list)
    for inst in insts:
        if inst.opname == 'LOAD_FAST':
            name = bytecode.co_varnames[inst.arg]
            rdnames[name].append(inst.offset)
        elif inst.opname == 'STORE_FAST':
            name = bytecode.co_varnames[inst.arg]
            wrnames[name].append(inst.offset)
    return rdnames, wrnames


def separate_loops(bytecode, outer, loops):
    """
    Separate top-level loops from the function

    Stores loopless instructions from the original function into `outer`.
    Stores list of loop instructions into `loops`.
    Both `outer` and `loops` are list-like (`append(item)` defined).
    """
    # XXX When an outer loop is rejected, there may be an inner loop
    # which would still allow lifting.
    endloop = None
    cur = None
    for inst in bytecode:
        if endloop is None:
            if inst.opname == 'SETUP_LOOP':
                cur = [inst]
                # Python may set the end of loop to the final jump destination
                # when nested in a if-else.  We need to scan the bytecode to
                # find the actual end of loop
                endloop = _scan_real_end_loop(bytecode, inst)
            else:
                outer.append(inst)
        else:
            cur.append(inst)
            if inst.next == endloop:
                for inst in cur:
                    if inst.opname in ['RETURN_VALUE', 'YIELD_VALUE',
                                       'BREAK_LOOP']:
                        # Reject if return, yield or break inside loop
                        outer.extend(cur)
                        break
                else:
                    loops.append(cur)
                endloop = None


def _scan_real_end_loop(bytecode, setuploop_inst):
    """Find the end of loop.
    Return the instruction offset.
    """
    start = setuploop_inst.next
    end = start + setuploop_inst.arg
    offset = start
    depth = 0
    while offset < end:
        inst = bytecode[offset]
        depth += inst.block_effect
        if depth < 0:
            return inst.next
        offset = inst.next

