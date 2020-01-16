
================
Notes on Hashing
================

Numba supports the built-in :func:`hash` and does so by simply calling the
:func:`__hash__` member function on the supplied argument. This makes it
trivial to add hash support for new types as all that is required is the
application of the extension API :func:`overload_method` decorator to overload
a function for computing the hash value for the new type registered to the
type's :func:`__hash__` method. For example::

    from numba.extending import overload_method

    @overload_method(myType, '__hash__')
    def myType_hash_overload(obj):
        # implementation details


The Implementation
==================

The implementation of the Numba hashing functions strictly follows that of
Python 3. The only exception to this is that for hashing Unicode and bytes (for
content longer than ``sys.hash_info.cutoff``) the only supported algorithm is
``siphash24`` (default in CPython 3). As a result Numba will match Python 3
hash values for all supported types under the default conditions described.

Unicode hash cache differences
------------------------------

Both Numba and CPython Unicode string internal representations have a ``hash``
member for the purposes of caching the string's hash value. This member is
always checked ahead of computing a hash value the with view of simply providing
a value from cache as it is considerably cheaper to do so. The Numba Unicode
string hash caching implementation behaves in a similar way to that of
CPython's. The only notable behavioral change (and its only impact is a minor
potential change in performance) is that Numba always computes and caches the
hash for Unicode strings created in ``nopython mode`` at the time they are boxed
for reuse in Python, this is too eager in some cases in comparison to CPython
which may delay hashing a new Unicode string depending on creation method. It
should also be noted that Numba copies in the ``hash`` member of the CPython
internal representation for Unicode strings when unboxing them to its own
representation so as to not recompute the hash of a string that already has a
hash value associated with it.

The accommodation of ``PYTHONHASHSEED``
---------------------------------------

The ``PYTHONHASHSEED`` environment variable can be used to seed the CPython
hashing algorithms for e.g. the purposes of reproduciblity. The Numba hashing
implementation directly reads the CPython hashing algorithms' internal state and
as a result the influence of ``PYTHONHASHSEED`` is replicated in Numba's
hashing implementations.
