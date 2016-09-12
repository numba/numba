======================
NBEP ?: Type Inference
======================

:Author: Siu Kwan Lam
:Date: Sept 2015
:Status: Draft


This document describes the current type inference implementation in numba.


Introduction
============

Numba uses type information to ensure that every variable in the user code can
be correctly lowered (translated into a low-level representation).  The type of
a variable describes the set of valid operations and available attributes.
Resolving this information during compilation avoids the overhead of type
checking and dispatching at runtime.  However, Python is dynamically typed and
the user does not declare variable types.  Since type information is absent.
we use type inference is to reconstruct the missing information.


Numba Type Semantic
===================

Type inference operates on Numba IR, a mostly static-single-assignment (SSA)
encoding of the Python bytecode.  Conceptually, all intermediate values in the
Python code is explicitly assigned to a variable in the IR.  Numba enforces
that each variable to have one type only.  A user variable (from the Python
source code) can be mapped to multiple variables in the IR.  They are *versions*
of a variable.  Each time a user variable is assigned to, a new version is
created.  From that point, all subsequent references will use the new version.
The user variable *evolves* as the function logic updates its type.  Merge
points (e.g. subsequent block to an if-else, the loop body, etc..) in the control
flow path need extra care.  Each incoming paths could be holding a different
variable version.  A new version is implicitly created at the merge point.
The incoming versions are also merged.  This may translate to an implicit cast.

Numba uses function overloading to emulate Python duck-typing.  The type of a
function can contain multiple call signatures that accept different argument
types and yields different return types.  The process to decide the best
signature for an overloaded function is called overload resolution.
Numba partially implements the C++  overload resolution scheme
(ISOCPP 13.3 Overload Resolution).  The scheme uses a "best fit" algorithm by
ranking each argument symmetrically.  The five possible rankings in increasing
order of penalty are:

* *Exact*: the expected type is the same as the actual type.
* *Promotion*: the actual type can be upcasted to the expected type by extending
  the precision without changing the behavior.
* *Safe conversion*: the actual type can be cast to the expected type by changing
  the type without losing information.
* *Unsafe conversion*: the actual type can be cast to the expected type by
  changing the type or downcast the type even if it is imprecise.
* *No match*: no valid operation can convert the actual type to the expected type.

It is possible to have an ambiguous resolution.  For example, a function with
signatures ``(int16, int32)`` and ``(int32, int16)`` can become ambiguous if
presented with the call types ``(int32, int32)``, because demoting either
argument to ``int16`` is equally "fit".  Fortunately, numba can usually resolve
such ambiguity by compiling a new version with the exact signature
``(int32, int32)``.

Type Inference
==============

The type inference in numba has three important components---type
variable, constraint network, and typing context.

* The *typing context* provides all the type information and typing related
  operations, including the logic for type unification, and the logic to typing
  of global and constant values.  It defines the semantic of the language that
  can be compiled by numba.

* A *type variable* holds the type of each variable (in the Numba IR).
  Conceptually, it is initialized to the universal type.  As it is re-assigned,
  it stores a common type by unifying the new type with the existing type.  The
  common type must be able to represent values of the new type and the existing
  type.  Type conversion is applied as necessary and precision loss is
  accepted for usability reason.

* The *constraint network* is a dependency graph built from the user code.  Each
  node represents an operation in the Numba IR and updates at least one type
  variables.  There may be cycles due to loops in user code.

The type inference process starts by seeding the argument types.  These initial
types are propagated in the constraint network, which eventually fills all the
type variables.  Due to cycles in the network, the process repeats until all
type variables converge or it fails with undecidable types.

A failure in type inference can be caused by two reasons.  One reason is user
error due to incorrect use of a type.  This type of error will also trigger an
exception in regular python execution.  Another reason is due to the use of an
unsupported feature, but the code is otherwise valid in regular python
execution.

Since functions can be overloaded, the type inference needs to decide the call
type, the calling signature, at each call site.  To decide the call type, the
overload resolution is applied to all known versions of the callee function.
These versions are described in *call-templates*.  A call-template can either be
concrete or abstract.  A concrete call-template defines a fixed list of all
possible signatures.  An abstract call-template defines the logic to compute
the accepted signature.  It is used to implement generic functions.

Numba-compiled functions are generic functions due to their ability to compile
new versions.  When it sees a new set of argument types, it triggers type
inference to validate and determine the return type. When there are nested calls
for numba-compiled functions, each call-site triggers type inference.
This poses a problem to recursive functions because the type inference will also
be triggered recursively.  Currently, simple single recursion is supported if
the signature is user-annotated by the user, which avoids unbound recursion in
type inference that will never terminate.
