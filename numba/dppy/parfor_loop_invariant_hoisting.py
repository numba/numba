from __future__ import print_function, division, absolute_import

def add_to_def_once_sets(a_def, def_once, def_more):
    '''If the variable is already defined more than once, do nothing.
       Else if defined exactly once previously then transition this
       variable to the defined more than once set (remove it from
       def_once set and add to def_more set).
       Else this must be the first time we've seen this variable defined
       so add to def_once set.
    '''
    if a_def in def_more:
        pass
    elif a_def in def_once:
        def_more.add(a_def)
        def_once.remove(a_def)
    else:
        def_once.add(a_def)

def compute_def_once_block(block, def_once, def_more, getattr_taken, typemap, module_assigns):
    '''Effect changes to the set of variables defined once or more than once
       for a single block.
       block - the block to process
       def_once - set of variable names known to be defined exactly once
       def_more - set of variable names known to be defined more than once
       getattr_taken - dict mapping variable name to tuple of object and attribute taken
       module_assigns - dict mapping variable name to the Global that they came from
    '''
    # The only "defs" occur in assignments, so find such instructions.
    assignments = block.find_insts(ir.Assign)
    # For each assignment...
    for one_assign in assignments:
        # Get the LHS/target of the assignment.
        a_def = one_assign.target.name
        # Add variable to def sets.
        add_to_def_once_sets(a_def, def_once, def_more)

        rhs = one_assign.value
        if isinstance(rhs, ir.Global):
            # Remember assignments of the form "a = Global(...)"
            # Is this a module?
            if isinstance(rhs.value, pytypes.ModuleType):
                module_assigns[a_def] = rhs.value.__name__
        if isinstance(rhs, ir.Expr) and rhs.op == 'getattr' and rhs.value.name in def_once:
            # Remember assignments of the form "a = b.c"
            getattr_taken[a_def] = (rhs.value.name, rhs.attr)
        if isinstance(rhs, ir.Expr) and rhs.op == 'call' and rhs.func.name in getattr_taken:
            # If "a" is being called then lookup the getattr definition of "a"
            # as above, getting the module variable "b" (base_obj)
            # and the attribute "c" (base_attr).
            base_obj, base_attr = getattr_taken[rhs.func.name]
            if base_obj in module_assigns:
                # If we know the definition of the module variable then get the module
                # name from module_assigns.
                base_mod_name = module_assigns[base_obj]
                if not is_const_call(base_mod_name, base_attr):
                    # Calling a method on an object could modify the object and is thus
                    # like a def of that object.  We call is_const_call to see if this module/attribute
                    # combination is known to not modify the module state.  If we don't know that
                    # the combination is safe then we have to assume there could be a modification to
                    # the module and thus add the module variable as defined more than once.
                    add_to_def_once_sets(base_obj, def_once, def_more)
            else:
                # Assume the worst and say that base_obj could be modified by the call.
                add_to_def_once_sets(base_obj, def_once, def_more)
        if isinstance(rhs, ir.Expr) and rhs.op == 'call':
            # If a mutable object is passed to a function, then it may be changed and
            # therefore can't be hoisted.
            # For each argument to the function...
            for argvar in rhs.args:
                # Get the argument's type.
                if isinstance(argvar, ir.Var):
                    argvar = argvar.name
                avtype = typemap[argvar]
                # If that type doesn't have a mutable attribute or it does and it's set to
                # not mutable then this usage is safe for hoisting.
                if getattr(avtype, 'mutable', False):
                    # Here we have a mutable variable passed to a function so add this variable
                    # to the def lists.
                    add_to_def_once_sets(argvar, def_once, def_more)

def compute_def_once_internal(loop_body, def_once, def_more, getattr_taken, typemap, module_assigns):
    '''Compute the set of variables defined exactly once in the given set of blocks
       and use the given sets for storing which variables are defined once, more than
       once and which have had a getattr call on them.
    '''
    # For each block...
    for label, block in loop_body.items():
        # Scan this block and effect changes to def_once, def_more, and getattr_taken
        # based on the instructions in that block.
        compute_def_once_block(block, def_once, def_more, getattr_taken, typemap, module_assigns)
        # Have to recursively process parfors manually here.
        for inst in block.body:
            if isinstance(inst, parfor.Parfor):
                # Recursively compute for the parfor's init block.
                compute_def_once_block(inst.init_block, def_once, def_more, getattr_taken, typemap, module_assigns)
                # Recursively compute for the parfor's loop body.
                compute_def_once_internal(inst.loop_body, def_once, def_more, getattr_taken, typemap, module_assigns)

def compute_def_once(loop_body, typemap):
    '''Compute the set of variables defined exactly once in the given set of blocks.
    '''
    def_once = set()   # set to hold variables defined exactly once
    def_more = set()   # set to hold variables defined more than once
    getattr_taken = {}
    module_assigns = {}
    compute_def_once_internal(loop_body, def_once, def_more, getattr_taken, typemap, module_assigns)
    return def_once

def find_vars(var, varset):
    assert isinstance(var, ir.Var)
    varset.add(var.name)
    return var

def _hoist_internal(inst, dep_on_param, call_table, hoisted, not_hoisted,
                    typemap, stored_arrays):
    if inst.target.name in stored_arrays:
        not_hoisted.append((inst, "stored array"))
        if config.DEBUG_ARRAY_OPT >= 1:
            print("Instruction", inst, " could not be hoisted because the created array is stored.")
        return False

    uses = set()
    visit_vars_inner(inst.value, find_vars, uses)
    diff = uses.difference(dep_on_param)
    if config.DEBUG_ARRAY_OPT >= 1:
        print("_hoist_internal:", inst, "uses:", uses, "diff:", diff)
    if len(diff) == 0 and is_pure(inst.value, None, call_table):
        if config.DEBUG_ARRAY_OPT >= 1:
            print("Will hoist instruction", inst, typemap[inst.target.name])
        hoisted.append(inst)
        if not isinstance(typemap[inst.target.name], types.npytypes.Array):
            dep_on_param += [inst.target.name]
        return True
    else:
        if len(diff) > 0:
            not_hoisted.append((inst, "dependency"))
            if config.DEBUG_ARRAY_OPT >= 1:
                print("Instruction", inst, " could not be hoisted because of a dependency.")
        else:
            not_hoisted.append((inst, "not pure"))
            if config.DEBUG_ARRAY_OPT >= 1:
                print("Instruction", inst, " could not be hoisted because it isn't pure.")
    return False

def find_setitems_block(setitems, itemsset, block, typemap):
    for inst in block.body:
        if isinstance(inst, ir.StaticSetItem) or isinstance(inst, ir.SetItem):
            setitems.add(inst.target.name)
            # If we store a non-mutable object into an array then that is safe to hoist.
            # If the stored object is mutable and you hoist then multiple entries in the
            # outer array could reference the same object and changing one index would then
            # change other indices.
            if getattr(typemap[inst.value.name], "mutable", False):
                itemsset.add(inst.value.name)
        elif isinstance(inst, parfor.Parfor):
            find_setitems_block(setitems, itemsset, inst.init_block, typemap)
            find_setitems_body(setitems, itemsset, inst.loop_body, typemap)

def find_setitems_body(setitems, itemsset, loop_body, typemap):
    """
      Find the arrays that are written into (goes into setitems) and the
      mutable objects (mostly arrays) that are written into other arrays
      (goes into itemsset).
    """
    for label, block in loop_body.items():
        find_setitems_block(setitems, itemsset, block, typemap)

def hoist(parfor_params, loop_body, typemap, wrapped_blocks):
    dep_on_param = copy.copy(parfor_params)
    hoisted = []
    not_hoisted = []

    # Compute the set of variable defined exactly once in the loop body.
    def_once = compute_def_once(loop_body, typemap)
    (call_table, reverse_call_table) = get_call_table(wrapped_blocks)

    setitems = set()
    itemsset = set()
    find_setitems_body(setitems, itemsset, loop_body, typemap)
    dep_on_param = list(set(dep_on_param).difference(setitems))
    if config.DEBUG_ARRAY_OPT >= 1:
        print("hoist - def_once:", def_once, "setitems:",
              setitems, "itemsset:", itemsset, "dep_on_param:",
              dep_on_param, "parfor_params:", parfor_params)

    for label, block in loop_body.items():
        new_block = []
        for inst in block.body:
            if isinstance(inst, ir.Assign) and inst.target.name in def_once:
                if _hoist_internal(inst, dep_on_param, call_table,
                                   hoisted, not_hoisted, typemap, itemsset):
                    # don't add this instruction to the block since it is
                    # hoisted
                    continue
            elif isinstance(inst, parfor.Parfor):
                new_init_block = []
                if config.DEBUG_ARRAY_OPT >= 1:
                    print("parfor")
                    inst.dump()
                for ib_inst in inst.init_block.body:
                    if (isinstance(ib_inst, ir.Assign) and
                        ib_inst.target.name in def_once):
                        if _hoist_internal(ib_inst, dep_on_param, call_table,
                                           hoisted, not_hoisted, typemap, itemsset):
                            # don't add this instuction to the block since it is hoisted
                            continue
                    new_init_block.append(ib_inst)
                inst.init_block.body = new_init_block

            new_block.append(inst)
        block.body = new_block
    return hoisted, not_hoisted

