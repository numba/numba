# Declare that function `myfunc` is going to be overloaded (have a
# substitutable Numba implementation)
@overload(myfunc)
# Define the overload function with formal arguments
# these arguments must be matched in the inner function implementation
def jit_myfunc(arg0, arg1, arg2, ...):
    # This scope is for typing, access is available to the *type* of all
    # arguments. This information can be used to change the behaviour of the
    # implementing function and check that the types are actually supported
    # by the implementation.

    print(arg0) # this will show the Numba type of arg0

    # This is the definition of the function that implements the `myfunc` work.
    # It does whatever algorithm is needed to implement myfunc.
    def myfunc_impl(arg0, arg1, arg2, ...): # match arguments to jit_myfunc
        # < Implementation goes here >
        return # whatever needs to be returned by the algorithm

    # return the implementation
    return myfunc_impl
