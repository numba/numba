from __future__ import print_function, division, absolute_import

from numba.core import ir
from numba.core.ir_utils import dead_code_elimination, simplify_CFG


def _run_inliner(func_ir, sig, template, arg_typs, expr, i, py_func, block,
                 work_list, typemap, calltypes, typingctx):
    from numba.core.inline_closurecall import (inline_closure_call,
                                          callee_ir_validator)

    # pass is typed so use the callee globals
    inline_closure_call(func_ir, py_func.__globals__,
                        block, i, py_func, typingctx=typingctx,
                        arg_typs=arg_typs,
                        typemap=typemap,
                        calltypes=calltypes,
                        work_list=work_list,
                        replace_freevars=False,
                        callee_validator=callee_ir_validator)
    return True


def _inline(func_ir, work_list, block, i, expr, py_func, typemap, calltypes,
            typingctx):
    # try and get a definition for the call, this isn't always possible as
    # it might be a eval(str)/part generated awaiting update etc. (parfors)
    to_inline = None
    try:
        to_inline = func_ir.get_definition(expr.func)
    except Exception:
        return False

    # do not handle closure inlining here, another pass deals with that.
    if getattr(to_inline, 'op', False) == 'make_function':
        return False

    # check this is a known and typed function
    try:
        func_ty = typemap[expr.func.name]
    except KeyError:
        return False
    if not hasattr(func_ty, 'get_call_type'):
        return False

    sig = calltypes[expr]
    is_method = False

    templates = getattr(func_ty, 'templates', None)
    arg_typs = sig.args

    if templates is None:
        return False

    assert(len(templates) == 1)

    # at this point we know we maybe want to inline something and there's
    # definitely something that could be inlined.
    return _run_inliner(
        func_ir, sig, templates[0], arg_typs, expr, i, py_func, block,
        work_list, typemap, calltypes, typingctx
    )


def _is_dufunc_callsite(expr, block):
    if expr.op == 'call':
        call_node = block.find_variable_assignment(expr.func.name).value
        # due to circular import we can not import DUFunc, TODO: Fix it
        if(call_node.value.__class__.__name__ == "DUFunc"):
            return call_node
    return None


def dufunc_inliner(func_ir, calltypes, typemap, typingctx):
    _DEBUG = False
    modified = False

    if _DEBUG:
        print('GUFunc before inlining DUFunc'.center(80, '-'))
        print(func_ir.dump())

    work_list = list(func_ir.blocks.items())
    # use a work list, look for call sites via `ir.Expr.op == call` and
    # then pass these to `self._do_work` to make decisions about inlining.
    while work_list:
        label, block = work_list.pop()
        for i, instr in enumerate(block.body):

            if isinstance(instr, ir.Assign):
                expr = instr.value
                if isinstance(expr, ir.Expr):
                    call_node = _is_dufunc_callsite(expr, block)
                    if call_node:
                        py_func = call_node.value._dispatcher.py_func
                        workfn = _inline(func_ir, work_list, block, i, expr,
                                         py_func, typemap, calltypes, typingctx)
                        if workfn:
                            modified = True
                            break  # because block structure changed
                    else:
                        continue
    if _DEBUG:
        print('GUFunc after inlining DUFunc'.center(80, '-'))
        print(func_ir.dump())
        print(''.center(80, '-'))

    if modified:
        # clean up leftover load instructions. This step is needed or else
        # SpirvLowerer would complain
        dead_code_elimination(func_ir, typemap=typemap)
        # clean up unconditional branches that appear due to inlined
        # functions introducing blocks
        func_ir.blocks = simplify_CFG(func_ir.blocks)

    if _DEBUG:
        print('GUFunc after inlining DUFunc, DCE, SimplyCFG'.center(80, '-'))
        print(func_ir.dump())
        print(''.center(80, '-'))

    return True
