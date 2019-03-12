"""
NRT specific optimizations
"""
from collections import defaultdict, deque


def remove_redundant_nrt_refct(ll_module):
    """
    Remove redundant reference count operations from the
    `llvmlite.binding.ModuleRef`. Remove the unnecessary nrt refct pairs
    within each block.  Decref calls are moved to the end of each block, just
    before the terminator.

    Note: non-threadsafe due to usage of global LLVMcontext
    """
    try:
        ll_module.get_function('NRT_incref')
    except NameError:
        return ll_module

    ll_module = ll_module.clone()
    defined_fns = (fn for fn in ll_module.functions
                   if not fn.is_declaration)

    for fn in defined_fns:
        if not fn.name.startswith('NRT_'):
            for bb in fn.blocks:
                _rewrite_block(bb)

    return ll_module


def _iter_refct_inst(bb):
    """Sequentially iterate over the instructions in *bb* and yielding
    the refcount instructions.

    Yields
    ------
    (inst, (callee, args))
    """
    for inst in bb.instructions:
        is_refop, inst, more = _process_inst(inst)
        if is_refop:
            yield inst, more


def _process_inst(inst):
    """Parse the instruction to determine if it is a refcount instruction.

    Returns
    -------
    (is_refop, inst, payload)
        where *payload* is *(callee, args)* if *is_refop* eval to True.
    """
    is_refop = False
    payload = ()
    if inst.opcode == 'call':
        operands = list(inst.operands)
        callee = operands[-1]
        args = operands[:-1]
        if callee.name in {'NRT_decref', 'NRT_incref'}:
            payload = callee, args
            is_refop = True
    return is_refop, inst, payload


def _rewrite_block(bb):
    """Remove extra refcount operations in the basicblock.
    """
    increfs = defaultdict(deque)
    decrefs = defaultdict(deque)
    to_remove = []

    # Track the increfs and decrefs
    for inst, (callee, args) in _iter_refct_inst(bb):
        [var] = args

        # Can't get rid of text manipulation fully yet.
        # The llvmlite API is not enough to tell what the operand is.
        operand_text = str(args[0])
        if var.name:
            varname = var.name
        elif operand_text == 'i8* null':
            varname = None
        else:
            # We have an assignment.  Take the LHS.
            varname = operand_text.split('=', 1)[0].strip()

        if varname:
            if callee.name == 'NRT_incref':
                increfs[varname].append(inst)
            elif callee.name == 'NRT_decref':
                decrefs[varname].append(inst)
        else:
            to_remove.append(inst)

    # Move decrefs down to just before the terminator
    term = list(bb.instructions)[-1]
    for declist in decrefs.values():
        for dec in declist:
            dec.move_before(term)

    # Drop pairs of inc/decrefs
    for k in increfs:
        while increfs[k] and decrefs[k]:
            inc = increfs[k].pop()
            dec = decrefs[k].popleft()
            to_remove.append(inc)
            to_remove.append(dec)

    # Actually remove the instructions
    for inst in to_remove:
        inst.erase_from_parent()

