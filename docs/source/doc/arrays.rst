*******************
Arrays
*******************
Numba has support for fast indexing and understands NumPy arrays and infers
types for various calls of the NumPy API.


Limitations
-------------
Unfortunately, there are a few pitfalls. We hope to resolve these in the
future, and to document them in the meantime:

=============================   =============================
Operation                       Example
=============================   =============================
Boundschecking                  ``array[N]``, with N < 0 or N > array.shape[0]

Wraparound                      ``array[-1]``

Calls to imported functions     Importing things from ``numpy``

                                ::

                                    from numpy import zeros

                                    @autojit
                                    def func():
                                        array = zeros(...)

Calling without a dtype         Calling ``zeros``, ``ones`` or ``empty``
                                without a dtype or with lists

                                ::

                                    np.zeros((M, N))                  # No dtype!
                                    np.zeros([M, N], dtype=np.double) # Not a tuple or int!

=============================   =============================


