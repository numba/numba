
=======================
Polymorphic dispatching
=======================

Functions compiled using :func:`~numba.jit` or :func:`~numba.vectorize`
are open-ended: they can be called with many different input types and
have to select (possibly compile) the right low-level specialization.
This document explains how this is done.


Requirements
============

JIT-compiled functions can take several arguments and each of them is
taken into account when selecting a specialization.  Thus it is a
form of multiple dispatch, more complex than single dispatching.

Each argument weighs into the selection based on its :ref:`Numba type
<numba-types>`.  Numba types are often more granular than Python types:
for example, Numba types Numpy arrays differently depending on their
dimensionality and their layout (C-contiguous, etc.).

Once a Numba type is inferred for each argument, a specialization must
be chosen amongst the available ones; or, if not suitable specialization
is found, a new one must be compiled.  This is not a trivial decision:
there can be multiple specializations compatible with a given concrete
signature (for example, say a two-argument function has compiled
specializations for ``(float64, float64)`` and ``(complex64, complex64)``,
and it is called with ``(float32, float32)``).

Therefore, we see there are two crucial steps in the dispatch mechanism:

1. infer the Numba types of the concrete arguments
2. select the best available specialization (or choose to compile a new one)
   for the inferred Numba types

Compile-time vs. run-time
-------------------------

This document discusses dispatching when it is done at runtime, i.e.
when a JIT-compiled function is called from pure Python.  In this context,
performance is important.  To stay in the realm of normal function call
overhead in Python, the overhead of dispatching should stay under the
microsecond.  Of course, *the faster the better*...

When a JIT-compiled function is called from another JIT-compiled
function (in :term:`nopython mode`), the polymorphism is resolved at
compile-time, using a non-performance critical mechanism, bearing zero
runtime performance overhead.

.. note::
   In practice, the performance-critical parts described here are coded in C.


Type resolution
===============

The first step is therefore to infer, at call-time, a Numba type for each
of the function's concrete arguments.  Given the finer granularity of
Numba types compared to Python types, one cannot simply lookup an object's
class and key a dictionary with it to obtain the corresponding Numba type.

Instead, there is a machinery to inspect the object and, based on its
Python type, query varying properties to infer the appropriate Numba
type.  This can be more or less complex: for example, a Python ``int``
argument will always infer to a Numba ``intp`` (a pointer-sized integer),
but a Python ``tuple`` argument can infer to multiple Numba types (depending
on the tuple's size and the concrete type of each of its elements).

The Numba type system is high-level and written in pure Python; there is
a pure Python machinery, based on a generic function, to do said inference
(in :mod:`numba.typing.typeof`).
That machinery is used for compile-time inference, e.g. on constants.
Unfortunately, it is too slow for run-time value-based dispatching.
It is only used as a fallback for rarely used (or difficult to infer)
types, and produces multiple-microsecond overhead.

Typecodes
---------

The Numba type system is really too high-level to be manipulated efficiently
from C code.  Therefore, the C dispatching layer uses another representation
based on integer typecodes.  Each Numba type gets a unique integer typecode
when constructed; also, an interning system ensure no two instances of same
type are created.  The dispatching layer is therefore able to *eschew*
the overhead of the Numba type system by working with simple integer
typecodes, amenable to well-known optimizations (fast hash tables, etc.).

The goal of the type resolution step becomes: infer a Numba *typecode*
for each of the function's concrete arguments.  Ideally, it doesn't deal
with Numba types anymore...

Hard-coded fast paths
---------------------

While eschewing the abstraction and object-orientation overhead of the type
system, the integer typecodes still have the same conceptual complexity.
Therefore, an important technique to speed up inference is to first go
through checks for the most important types, and hard-code a fast resolution
for each of them.

Several types benefit from such an optimization, notably:

* the most basic Python scalars (``bool``, ``int``, ``float``, ``complex``)
* some basic Numpy scalars (the various kinds of integer, floating-point,
  complex numbers)
* Numpy arrays of certain dimensionalities and basic element types

Each of those fast paths ideally uses a hard-coded result value or a direct
table lookup after a few simple checks.

However, we can't apply that technique to all argument types; there would
be an explosion of ad-hoc internal caches, and it would become difficult to
maintain.  Besides, the recursive application of hard-coded fast paths
would not necessarily combine into a low overhead (in the nested tuple
case, for example).

Fingerprint-based typecode cache
--------------------------------

For non-so-trivial types (imagine a tuple, or a Numpy ``datetime64`` array,
for example), the hard-coded fast paths don't match.  Another mechanism
then kicks in, more generic.

The principle here is to examine each argument value, as the pure Python
machinery would do, and to describe its Numba type unambiguously.  The
difference is that *we don't actually compute a Numba type*.  Instead, we
compute a simple bytestring, a low-level possible denotation of that
Numba type: a *fingerprint*.  The fingerprint format is designed to be
short and extremely simple to compute from C code (in practice, it has
a bytecode-like format).

Once the fingerprint is computed, it is looked up in a cache mapping
fingerprints to typecodes.  The cache is a hash table, and the lookup
is fast thanks to the fingerprints being generally very short (rarely
more than 20 bytes).

If the cache lookup fails, the typecode must first be computed using the
slow pure Python machinery.  Luckily, this would only happen once: on
subsequent calls, the cached typecode would be returned for the given
fingerprint.

In rare cases, a fingerprint cannot be computed efficiently.  This is
the case for some types which cannot be easily inspected from C: for
example ``cffi`` function pointers.  Then, the slow Pure Python machinery
is invoked at each function call with such an argument.

.. note::
   Two fingerprints may denote a single Numba type.  This does not make
   the mechanism incorrect; the cache would just be less efficient.


Summary
-------

...
