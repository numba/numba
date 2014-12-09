"""
Macro handling passes

Macros are expanded on block-by-block
"""
from __future__ import absolute_import, print_function, division
from numba import ir


class MacroError(Exception):
    '''
    An exception thrown during macro expansion
    '''
    pass


def expand_macros(blocks):
    '''
    Performs macro expansion on blocks

    Args
    ----
    blocks: list
        the blocks to macro-expand
    return: bool
        True if any macros were expanded
    '''
    constants = {}
    expanded = False
    for blk in blocks.values():
        module_getattr_folding(constants, blk)
        block_expanded = expand_macros_in_block(constants, blk)
        expanded = expanded or block_expanded
    return expanded

def module_getattr_folding(constants, block):
    '''
    Performs constant-folding of getattr instructions within a block. Any
    constants defined within the block are also added to the constant pool.

    Args
    ----
    constants: dict
        The pool of constants to use, which will be updated with any new
        constants in this block
    block: ir.Block
        The block to perform constant folding on
    '''
    for inst in block.body:
        if isinstance(inst, ir.Assign):
            rhs = inst.value

            if isinstance(rhs, ir.Global):
                constants[inst.target.name] = rhs.value

            elif isinstance(rhs, ir.Expr):
                if rhs.op == 'getattr':
                    if rhs.value.name in constants:
                        base = constants[rhs.value.name]
                        constants[inst.target.name] = getattr(base, rhs.attr)

                elif rhs.op == 'build_tuple':
                    if all(i.name in constants for i in rhs.items):
                        tupk = tuple(constants[i.name] for i in rhs.items)
                        constants[inst.target.name] = tupk

                elif rhs.op == 'build_list':
                    if all(i.name in constants for i in rhs.items):
                        tupk = list(constants[i.name] for i in rhs.items)
                        constants[inst.target.name] = tupk

            elif isinstance(rhs, ir.Const):
                constants[inst.target.name] = rhs.value

            elif isinstance(rhs, ir.Var) and rhs.name in constants:
                constants[inst.target.name] = constants[rhs.name]

            elif isinstance(rhs, ir.FreeVar):
                constants[inst.target.name] = rhs.value

def expand_macros_in_block(constants, block):
    '''
    Performs macro expansion on a block.

    Args
    ----
    constants: dict
        The pool of constants which contains the values which contains mappings
        from variable names to callee names
    block: ir.Block
        The block to perform macro expansion on
    return: bool
        True if any macros were expanded
    '''
    expanded = False
    for inst in block.body:
        if isinstance(inst, ir.Assign):
            rhs = inst.value
            if isinstance(rhs, ir.Expr) and rhs.op == 'call':
                callee = rhs.func
                macro = constants.get(callee.name)
                if isinstance(macro, Macro):
                    # Rewrite calling macro
                    assert macro.callable
                    args = [constants[arg.name] for arg in rhs.args]

                    kws = {}
                    for k, v in rhs.kws:
                        if v.name in constants:
                            kws[k] = constants[v.name]

                        else:
                            msg = "Argument {name!r} must be a " \
                                  "constant at {loc}".format(name=k,
                                                             loc=inst.loc)
                            raise ValueError(msg)

                    try:
                        result = macro.func(*args, **kws)
                    except BaseException as e:
                        msg = str(e)
                        headfmt = "Macro expansion failed at {line}"
                        head = headfmt.format(line=inst.loc)
                        newmsg = "{0}:\n{1}".format(head, msg)
                        raise MacroError(newmsg)
                    if result:
                        # Insert a new function
                        result.loc = rhs.loc
                        inst.value = ir.Expr.call(func=result, args=rhs.args,
                                                  kws=rhs.kws, loc=rhs.loc)
                        expanded = True
            elif isinstance(rhs, ir.Expr) and rhs.op == 'getattr':
                # Rewrite get attribute to macro call
                # Non-calling macro must be triggered by get attribute
                base = constants.get(rhs.value.name)
                if base is not None:
                    value = getattr(base, rhs.attr)
                    if isinstance(value, Macro):
                        macro = value
                        if not macro.callable:
                            intr = ir.Intrinsic(macro.name, macro.func, args=())
                            inst.value = ir.Expr.call(func=intr, args=(),
                                                      kws=(), loc=rhs.loc)
                            expanded = True
    return expanded


class Macro(object):
    '''
    A macro object is expanded to a function call

    Args
    ----
    name: str
        Name of this Macro
    func: function
        Function that evaluates the macro expansion
    callable: bool
        True if the Macro represents a callable function.
        False if it is represents some other type.
    argnames: list
        If ``callable`` is True, this holds a list of the names of arguments
        to the function.
    '''

    __slots__ = 'name', 'func', 'callable', 'argnames'

    def __init__(self, name, func, callable=False, argnames=None):
        self.name = name
        self.func = func
        self.callable = callable
        self.argnames = argnames

    def __repr__(self):
        return '<macro %s -> %s>' % (self.name, self.func)

