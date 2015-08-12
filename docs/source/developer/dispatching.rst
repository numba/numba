
=======================
Polymorphic dispatching
=======================

Functions compiled using :func:`~numba.jit` or :func:`~numba.vectorize`
are open-ended: they can be called with many different input types and
have to select (possibly compile on-the-fly) the right low-level
specialization.  We hereby explain how this mechanism is implemented.


Requirements
============

JIT-compiled functions can take several arguments and each of them is
taken into account when selecting a specialization.  Thus it is a
form of multiple dispatch, more complex than single dispatch.

Each argument weighs in the selection based on its :ref:`Numba type
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

Therefore, there are two crucial steps in the dispatch mechanism:

1. infer the Numba types of the concrete arguments
2. select the best available specialization (or choose to compile a new one)
   for the inferred Numba types

Compile-time vs. run-time
-------------------------

This document discusses dispatching when it is done at runtime, i.e.
when a JIT-compiled function is called from pure Python.  In that context,
performance is important.  To stay in the realm of normal function call
overhead in Python, the overhead of dispatching should stay under a
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
Python type, query various properties to infer the appropriate Numba
type.  This can be more or less complex: for example, a Python ``int``
argument will always infer to a Numba ``intp`` (a pointer-sized integer),
but a Python ``tuple`` argument can infer to multiple Numba types (depending
on the tuple's size and the concrete type of each of its elements).

The Numba type system is high-level and written in pure Python; there is
a pure Python machinery, based on a generic function, to do said inference
(in :mod:`numba.typing.typeof`).  That machinery is used for compile-time
inference, e.g. on constants.  Unfortunately, it is too slow for run-time
value-based dispatching.  It is only used as a fallback for rarely used
(or difficult to infer) types, and exhibits multiple-microsecond overhead.

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

* basic Python scalars (``bool``, ``int``, ``float``, ``complex``);
* basic Numpy scalars (the various kinds of integer, floating-point,
  complex numbers);
* Numpy arrays of certain dimensionalities and basic element types.

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
   the mechanism incorrect; it only creates more cache entries.


Summary
-------

Type resolution of a function argument involves the following mechanisms
in order:

* Try a few hard-coded fast paths, for common simple types.
* If the above failed, compute a fingerprint for the argument and lookup
  its typecode in a cache.
* If all the above failed, invoke the pure Python machinery which will
  determine a Numba type for the argument (and look up its typecode).


Specialization selection
========================

At the previous step, an integer typecode has been determined for each
concrete argument to the JIT-compiled function.  Now it remains to match
that concrete signature against each of the available specializations for
the function.  There can be three outcomes:

* There is a satisfying best match: the corresponding specialization
  is then invoked (it will handle argument unboxing and other details).
* There is a tie between two or more "best matches": an exception is raised,
  refusing to solve the ambiguity.
* There is no satisfying match: a new specialization is compiled tailored
  for the concrete argument types that were inferred.

The selection works by looping over all available specializations, and
computing the compatibility of each concrete argument type with the
corresponding type in the specialization's intended signature.  Specifically,
we are interested in:

1. Whether the concrete argument type is allowed to convert implicitly to
   the specialization's argument type;
2. If so, at what semantic (user-visible) cost the conversion comes.

Implicit conversion rules
-------------------------

There are five possible kinds of implicit conversion from a source type
to a destination type (note this is an asymmetric relationship):

1. *exact match*: the two types are identical; this is the ideal case,
   since the specialization would behave exactly as intended;
2. *same-kind promotion*: the two types belong to the same "kind" (for
   example ``int32`` and ``int64`` are two integer types), and the source
   type can be converted losslessly to the destination type (e.g. from
   ``int32`` to ``int64``, but not the reverse);
3. *safe conversion*: the two types belong to different kinds, but the
   source type can be reasonably converted to the destination type
   (e.g. from ``int32`` to ``float64``, but not the reverse);
4. *unsafe conversion*: a conversion is available from the source type
   to the destination type, but it may lose precision, magnitude, or
   another desirable quality.
5. *no conversion*: there is no correct or reasonably efficient way to
   convert between the two types (for example between an ``int64`` and a
   ``datetime64``, or a C-contiguous array and a Fortran-contiguous array).

When a specialization is examined, the latter two cases eliminate it from
the final choice: i.e. when at least one argument has *no conversion* or
only an *unsafe conversion* to the signature's argument type.

.. note::
   However, if the function is compiled with explicit signatures
   in the :func:`~numba.jit` call (and therefore it is not allowed to compile
   new specializations), *unsafe conversion* is allowed.

Candidates and best match
-------------------------

If a specialization is not eliminated by the rule above, it enters the
list of *candidates* for the final choice.  Those candidates are ranked
by an ordered 4-uple of integers: ``(number of unsafe conversions,
number of safe conversions, number of same-kind promotions, number of
exact matches)`` (note the sum of the tuple's elements is equal to the
number of arguments).  The best match is then the #1 result in sorted
ascending order, thereby preferring exact matches over promotions,
promotions over safe conversions, safe conversions over unsafe conversions.

Implementation
--------------

The above-described mechanism works on integer typecodes, not on Numba
types.  It uses an internal hash table storing the possible conversion
kind for each pair of compatible types.  The internal hash table is in part
built at startup (for built-in trivial types such as ``int32``, ``int64``
etc.), in part filled dynamically (for arbitrarily complex types such
as array types: for example to allow using a C-contiguous 2D array where
a function expects a non-contiguous 2D array).

Summary
-------

Selecting the right specialization involves the following steps:

* Examine each available specialization and match it against the concrete
  argument types.
* Eliminate any specialization where at least one argument doesn't offer
  sufficient compatibility.
* If there are remaining candidates, choose the best one in terms of
  preserving the types' semantics.


Miscellaneous
=============

Some `benchmarks of dispatch performance
<https://github.com/numba/numba-benchmark/blob/master/benchmarks/bench_dispatch.py>`_
exist in the `Numba benchmarks <https://github.com/numba/numba-benchmark>`_
repository.

Some unit tests of specific aspects of the machinery are available
in :mod:`numba.tests.test_typeinfer` and :mod:`numba.tests.test_typeof`.
Higher-level dispatching tests are in :mod:`numba.tests.test_dispatcher`.
