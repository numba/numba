import numba
from numba.extending import overload
from numba import njit, types


def bar(x):
    """A function stub to overload"""
    pass


@overload(bar, inline='always')
def ol_bar_tuple(x):
    # An overload that will always inline, there is a type guard so that this
    # only applies to UniTuples.
    if isinstance(x, types.UniTuple):
        def impl(x):
            return x[0]
        return impl


def cost_model(expr, caller, callee):
    # Only inline if the type of the argument is an Integer
    return isinstance(caller.typemap[expr.args[0].name], types.Integer)


@overload(bar, inline=cost_model)
def ol_bar_scalar(x):
    # An overload that will inline based on a cost model, it only applies to
    # scalar values in the numerical domain as per the type guard on Number
    if isinstance(x, types.Number):
        def impl(x):
            return x + 1
        return impl


@njit
def foo():

    # This will resolve via `ol_bar_tuple` as the argument is a types.UniTuple
    # instance. It will always be inlined as specified in the decorator for this
    # overload.
    a = bar((1, 2, 3))

    # This will resolve via `ol_bar_scalar` as the argument is a types.Number
    # instance, hence the cost_model will be used to determine whether to
    # inline.
    # The function will be inlined as the value 100 is an IntegerLiteral which
    # is an instance of a types.Integer as required by the cost_model function.
    b = bar(100)

    # This will also resolve via `ol_bar_scalar` as the argument is a
    # types.Number instance, again the cost_model will be used to determine
    # whether to inline.
    # The function will not be inlined as the complex value is not an instance
    # of a types.Integer as required by the cost_model function.
    c = bar(300j)

    return a + b + c


foo()
