from numba import njit, ir
import numba


@njit(inline='never')
def never_inline():
    return 100


@njit(inline='always')
def always_inline():
    return 200


def sentinel_cost_model(expr, caller_info, callee_info):
    # this cost model will return True (i.e. do inlining) if either:
    # a) the callee IR contains an `ir.Const(37)`
    # b) the caller IR contains an `ir.Const(13)` logically prior to the call
    #    site

    # check the callee
    for blk in callee_info.blocks.values():
        for stmt in blk.body:
            if isinstance(stmt, ir.Assign):
                if isinstance(stmt.value, ir.Const):
                    if stmt.value.value == 37:
                        return True

    # check the caller
    before_expr = True
    for blk in caller_info.blocks.values():
        for stmt in blk.body:
            if isinstance(stmt, ir.Assign):
                if isinstance(stmt.value, ir.Expr):
                    if stmt.value == expr:
                        before_expr = False
                if isinstance(stmt.value, ir.Const):
                    if stmt.value.value == 13:
                        return True & before_expr
    return False


@njit(inline=sentinel_cost_model)
def maybe_inline1():
    # Will not inline based on the callee IR with the declared cost model
    # The following is ir.Const(300).
    return 300


@njit(inline=sentinel_cost_model)
def maybe_inline2():
    # Will inline based on the callee IR with the declared cost model
    # The following is ir.Const(37).
    return 37


@njit
def foo():
    a = never_inline()  # will never inline
    b = always_inline()  # will always inline

    # will not inline as the function does not contain a magic constant known to
    # the cost model, and the IR up to the call site does not contain a magic
    # constant either
    d = maybe_inline1()

    # declare this magic constant to trigger inlining of maybe_inline1 in a
    # subsequent call
    magic_const = 13

    # will inline due to above constant declaration
    e = maybe_inline1()

    # will inline as the maybe_inline2 function contains a magic constant known
    # to the cost model
    c = maybe_inline2()

    return a + b + c + d + e + magic_const


foo()
