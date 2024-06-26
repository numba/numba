.. _deprecation:

===================
Deprecation Notices
===================

This section contains information about deprecation of behaviours, features and
APIs that have become undesirable/obsolete. Any information about the schedule
for their deprecation and reasoning behind the changes, along with examples, is
provided. However, first is a small section on how to suppress deprecation
warnings that may be raised from Numba so as to prevent warnings propagating
into code that is consuming Numba.

.. _suppress_deprecation_warnings:

Suppressing Deprecation warnings
================================
All Numba deprecations are issued via ``NumbaDeprecationWarning`` or
``NumbaPendingDeprecationWarning`` s, to suppress the reporting of
these the following code snippet can be used::

    from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
    import warnings

    warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
    warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

The ``action`` used above is ``'ignore'``, other actions are available, see
`The Warnings Filter <https://docs.python.org/3/library/warnings.html#the-warnings-filter>`_
documentation for more information.

.. note:: It is **strongly recommended** that applications and libraries which
          choose to suppress these warnings should pin their Numba dependency
          to a suitable version because their users will no longer be aware of
          the coming incompatibility.

Deprecation of reflection for List and Set types
================================================
Reflection (:term:`reflection`) is the jargon used in Numba to describe the
process of ensuring that changes made by compiled code to arguments that are
mutable Python container data types are visible in the Python interpreter when
the compiled function returns. Numba has for some time supported reflection of
``list`` and ``set`` data types and it is support for this reflection that
is scheduled for deprecation with view to replace with a better implementation.

Reason for deprecation
----------------------
First recall that for Numba to be able to compile a function in ``nopython``
mode all the variables must have a concrete type ascertained through type
inference. In simple cases, it is clear how to reflect changes to containers
inside ``nopython`` mode back to the original Python containers. However,
reflecting changes to complex data structures with nested container types (for
example, lists of lists of integers) quickly becomes impossible to do
efficiently and consistently. After a number of years of experience with this
problem, it is clear that providing this behaviour is both fraught with
difficulty and often leads to code which does not have good performance (all
reflected data has to go through special APIs to convert the data to native
formats at call time and then back to CPython formats at return time). As a
result of this, the sheer number of reported problems in the issue tracker, and
how well a new approach that was taken with ``typed.Dict`` (typed dictionaries)
has gone, the core developers have decided to deprecate the noted ``reflection``
behaviour.


Example(s) of the impact
------------------------

At present only a warning of the upcoming change is issued. In future code such
as::

  from numba import njit

  @njit
  def foo(x):
      x.append(10)

  a = [1, 2, 3]
  foo(a)

will require adjustment to use a ``typed.List`` instance, this typed container
is synonymous to the :ref:`feature-typed-dict`. An example of translating the
above is::

    from numba import njit
    from numba.typed import List

    @njit
    def foo(x):
        x.append(10)

    a = [1, 2, 3]
    typed_a = List()
    [typed_a.append(x) for x in a]
    foo(typed_a)

For more information about ``typed.List`` see :ref:`feature-typed-list`. Further
usability enhancements for this feature were made in the 0.47.0 release
cycle.

Schedule
--------
This feature will be removed with respect to this schedule:

* Pending-deprecation warnings will be issued in version 0.44.0
* Prominent notice will be given for a minimum of two releases prior to full
  removal.

Recommendations
---------------
Projects that need/rely on the deprecated behaviour should pin their dependency
on Numba to a version prior to removal of this behaviour, or consider following
replacement instructions that will be issued outlining how to adjust to the
change.

Expected Replacement
--------------------
As noted above ``typed.List`` will be used to permit similar functionality to
reflection in the case of ``list`` s, a ``typed.Set`` will provide the
equivalent for ``set`` (not implemented yet!). The advantages to this approach
are:

* That the containers are typed means type inference has to work less hard.
* Nested containers (containers of containers of ...) are more easily
  supported.
* Performance penalties currently incurred translating data to/from native
  formats are largely avoided.
* Numba's ``typed.Dict`` will be able to use these containers as values.


Deprecation of :term:`object mode` `fall-back` behaviour when using ``@jit``
============================================================================

.. note::

    This feature was removed in 0.59.0, see the schedule section below.

The ``numba.jit`` decorator has for a long time followed the behaviour of first
attempting to compile the decorated function in :term:`nopython mode` and should
this compilation fail it will `fall-back` and try again to compile but this time
in :term:`object mode`. It is this `fall-back` behaviour which is being
deprecated, the result of which will be that ``numba.jit`` will by default
compile in :term:`nopython mode` and :term:`object mode` compilation will
become `opt-in` only.

.. note::

    It is relatively common for the ``numba.jit`` decorator to be used within
    other decorators to provide an easy path to compilation. Due to this change,
    deprecation warnings may be raised from such call sites. To avoid these
    warnings, it's recommended to either
    :ref:`suppress them <suppress_deprecation_warnings>` if the application does
    not rely on :term:`object mode` `fall-back` or to check the documentation
    for the decorator to see how to pass application appropriate options through
    to the wrapped ``numba.jit`` decorator. An example of this within the Numba
    API would be ``numba.vectorize``. This decorator simply forwards keyword
    arguments to the internal ``numba.jit`` decorator call site such that e.g.
    ``@vectorize(nopython=True)`` would be an appropriate declaration for a
    ``nopython=True`` mode use of ``@vectorize``.

Reason for deprecation
----------------------
The `fall-back` has repeatedly caused confusion for users as seemingly innocuous
changes in user code can lead to drastic performance changes as code which may
have once compiled in :term:`nopython mode` mode may silently switch to
compiling in :term:`object mode` e.g::

    from numba import jit

    @jit
    def foo():
        l = []
        for x in range(10):
            l.append(x)
        return l

    foo()

    assert foo.nopython_signatures # this was compiled in nopython mode

    @jit
    def bar():
        l = []
        for x in range(10):
            l.append(x)
        return reversed(l) # innocuous change, but no reversed support in nopython mode

    bar()

    assert not bar.nopython_signatures # this was not compiled in nopython mode

Another reason to remove the `fall-back` is that it is confusing for the
compiler engineers developing Numba as it causes internal state problems that
are really hard to debug and it makes manipulating the compiler pipelines
incredibly challenging.

Further, it has long been considered best practice that the
:term:`nopython mode` keyword argument in the ``numba.jit`` decorator is set to
``True`` and that any user effort spent should go into making code work in this
mode as there's very little gain if it does not. The result is that, as Numba
has evolved, the amount of use :term:`object mode` gets in practice and its
general utility has decreased. It can be noted that there are some minor
improvements available through the notion of :term:`loop-lifting`, the cases of
this being used in practice are, however, rare and often a legacy from use of
less-recent Numba whereby such behaviour was better accommodated/the use of
``@jit`` with `fall-back` was recommended.


Example(s) of the impact
------------------------
At present a warning of the upcoming change is issued if ``@jit`` decorated code
uses the `fall-back` compilation path. In future code such as::

    @jit
    def bar():
        l = []
        for x in range(10):
            l.append(x)
        return reversed(l)

    bar()

will simply not compile, a ``TypingError`` would be raised.

A further consequence of this change is that the ``nopython`` keyword argument
will become redundant as :term:`nopython mode` will be the default. As a result,
following this change, supplying the keyword argument as ``nopython=False`` will
trigger a warning stating that the implicit default has changed to ``True``.
Essentially this keyword will have no effect following removal of this feature.

Schedule
--------
This feature was removed with respect to this schedule:

* Deprecation warnings were issued in version 0.44.0.
* Prominent notice was given in 0.57.0.
* The feature was removed in 0.59.0.

Recommendations
---------------
Projects that need/rely on the deprecated behaviour should pin their dependency
on Numba to a version prior to removal of this behaviour.

General advice to accommodate the scheduled deprecation:

Users with code compiled at present with ``@jit`` can supply the
``nopython=True`` keyword argument, if the code continues to compile then the
code is already ready for this change. If the code does not compile, continue
using the ``@jit`` decorator without ``nopython=True`` and profile the
performance of the function. Then remove the decorator and again check the
performance of the function. If there is no benefit to having the ``@jit``
decorator present consider removing it! If there is benefit to having the
``@jit`` decorator present, then to be future proof supply the keyword argument
``forceobj=True`` to ensure the function is always compiled in
:term:`object mode`.

Advice for users of the "loop-lifting" feature:

If object mode compilation with loop-lifting is needed it should be
explicitly declared through supplying the keyword arguments ``forceobj=True``
and ``looplift=True`` to the ``@jit`` decorator.

Advice for users setting ``nopython=False``:

This is essentially specifying the implicit default prior to removal of this
feature, either remove the keyword argument or change the value to ``True``.



.. _deprecation-of-generated-jit:

Deprecation of ``generated_jit``
================================
The top level API function ``numba.generated_jit`` provides functionality that
allows users to write JIT compilable functions that have different
implementations based on the types of the arguments to the function. This is a
hugely useful concept and is also key to Numba's internal implementation.

Reason for deprecation
----------------------

There are a number of reasons for this deprecation.

First, ``generated_jit`` breaks the concept of "JIT transparency" in that if the
JIT compiler is disabled, the source code does not execute the same way as it
would were the JIT compiler present.

Second, internally Numba uses the ``numba.extending.overload`` family of
decorators to access an equivalent functionality to ``generated_jit``. The
``overload`` family of decorators are more powerful than ``generated_jit`` as
they support far more options and both the CPU and CUDA targets. Essentially a
replacement for ``generated_jit`` already exists and has been recommended and
preferred for a long while.

Third, the public extension API decorators are far better maintained than
``generated_jit``. This is an important consideration due to Numba's limited
resources, fewer duplicated pieces of functionality to maintain will reduce
pressure on these resources.

For more information on the ``overload`` family of decorators see the
:ref:`high level extension API documentation <high-level-extending>`.

Example(s) of the impact
------------------------

Any source code using ``generated_jit`` would fail to work once the
functionality has been removed.

Schedule
--------

This feature was removed with respect to this schedule:

* Deprecation warnings were issued in version 0.57.0.
* Removal took place in version 0.59.0.

Recommendations
---------------

Projects that need/rely on the deprecated behaviour should pin their dependency
on Numba to a version prior to removal of this behaviour, or consider following
replacement instructions below that outline how to adjust to the change.

Replacement
-----------

The ``overload`` decorator offers a replacement for the functionality available
through ``generated_jit``. An example follows of translating from one to the
other. First define a type specialised function dispatch with the
``generated_jit`` decorator::

  from numba import njit, generated_jit, types

  @generated_jit
  def select(x):
      if isinstance(x, types.Float):
          def impl(x):
              return x + 1
          return impl
      elif isinstance(x, types.UnicodeType):
          def impl(x):
              return x + " the number one"
          return impl
      else:
          raise TypeError("Unsupported Type")

  @njit
  def foo(x):
      return select(x)

  print(foo(1.))
  print(foo("a string"))

Conceptually, ``generated_jit`` is like ``overload``, but with ``generated_jit``
the overloaded function is the decorated function. Taking the example above and
adjusting it to use the ``overload`` API::

  from numba import njit, types
  from numba.extending import overload

  # A pure python implementation that will run if the JIT compiler is disabled.
  def select(x):
      if isinstance(x, float):
          return x + 1
      elif isinstance(x, str):
          return x + " the number one"
      else:
          raise TypeError("Unsupported Type")

  # An overload for the `select` function cf. generated_jit
  @overload(select)
  def ol_select(x):
      if isinstance(x, types.Float):
          def impl(x):
              return x + 1
          return impl
      elif isinstance(x, types.UnicodeType):
          def impl(x):
              return x + " the number one"
          return impl
      else:
          raise TypeError("Unsupported Type")

  @njit
  def foo(x):
      return select(x)

  print(foo(1.))
  print(foo("a string"))

Further, users that are using ``generated_jit`` to dispatch on some of the more
primitive types may find that Numba's support for ``isinstance`` is sufficient,
for example::

  @njit # NOTE: standard @njit decorator.
  def select(x):
      if isinstance(x, float):
          return x + 1
      elif isinstance(x, str):
          return x + " the number one"
      else:
          raise TypeError("Unsupported Type")

  @njit
  def foo(x):
      return select(x)

  print(foo(1.))
  print(foo("a string"))


.. _deprecation-numba-pycc:

Deprecation of the ``numba.pycc`` module
========================================
Numba has supported some degree of Ahead-of-Time (AOT) compilation through the
use of the tools in the ``numba.pycc`` module. This capability is very important
to the Numba project and following an assessment of the viability of the current
approach, it was decided to deprecate it in favour of developing new technology
to better meet current needs.

Reason for deprecation
----------------------

There are a number of reasons for this deprecation.

* ``numba.pycc`` tools create C-Extensions that have symbols that are only
  usable from the Python interpreter, they are not compatible with calls made
  from within code compiled using Numba's JIT compiler. This drastically reduces
  the utility of AOT compiled functions.
* ``numba.pycc`` has some reliance on ``setuptools`` (and ``distutils``) which
  is something Numba is trying to reduce, particularly due to the upcoming
  removal of ``distutils`` in Python 3.12.
* The ``numba.pycc`` compilation chain is very limited in terms of its feature
  set in comparison to Numba's JIT compiler, it also has numerous technical
  issues to do with declaring and linking both internal and external libraries.
* The number of users of ``numba.pycc`` is assumed to be quite small, this was
  indicated through discussions at a Numba public meeting on 2022-10-04 and
  issue #8509.
* The Numba project is working on new innovations in the AOT compiler space and
  the maintainers consider it a better use of resources to develop these than
  maintain and develop ``numba.pycc``.

Example(s) of the impact
------------------------

Any source code using ``numba.pycc`` would fail to work once the functionality
has been removed.

Schedule
--------

This feature will be removed with respect to this schedule:

* Pending-deprecation warnings will be issued in version 0.57.0.
* Deprecation warnings will be issued once a replacement is developed.
* Deprecation warnings will be given for a minimum of two releases prior to full
  removal.

Recommendations
---------------

Projects that need/rely on the deprecated behaviour should pin their dependency
on Numba to a version prior to removal of this behaviour, or consider following
replacement instructions below that outline how to adjust to the change.

Replacement
-----------

A replacement for this functionality is being developed as part of the Numba
2023 development focus. The ``numba.pycc`` module will not be removed until this
replacement functionality is able to provide similar utility and offer an
upgrade path. At the point of the new technology being deemed suitable,
replacement instructions will be issued.


Deprecation and removal of CUDA Toolkits < 11.2 and devices with CC < 5.0
=========================================================================

- Support for CUDA toolkits less than 11.2 has been removed.
- Support for devices with Compute Capability < 5.0 is deprecated and will be
  removed in the future.

Recommendations
---------------

- For devices of Compute Capability 3.0 and 3.2, Numba 0.55.1 or earlier will
  be required.
- CUDA toolkit 11.2 or later should be installed.

Schedule
--------

- In Numba 0.55.1: support for CC < 5.0 and CUDA toolkits < 10.2 was deprecated.
- In Numba 0.56: support for CC < 3.5 and CUDA toolkits < 10.2 was removed.
- In Numba 0.57: Support for CUDA toolkit 10.2 was removed.
- In Numba 0.58: Support CUDA toolkits 11.0 and 11.1 was removed.
- In a future release: Support for CC < 5.0 will be removed.

Deprecation of old-style ``NUMBA_CAPTURED_ERRORS``
==================================================

The use of ``NUMBA_CAPTURED_ERRORS=old_style`` environment variable is being 
deprecated in Numba.

Reason for deprecation
----------------------

Previously, this variable allowed controlling how Numba handles exceptions 
during compilation that do not inherit from ``numba.core.errors.NumbaError``. 
The default "old_style" behavior was to capture and wrap these errors, often 
obscuring the original exception.

The new "new_style" option treats non-``NumbaError`` exceptions as hard errors, 
propagating them without capturing. This differentiates compilation errors from 
unintended exceptions during compilation.

The old style will eventually be removed in favor of the new behavior. Users 
should migrate to setting ``NUMBA_CAPTURED_ERRORS='new_style'`` to opt-in to the 
new exception handling. This will become the default in the future.

Impact
------

The impact of this deprecation will only affect those who are extending Numba
functionality. 

Recommendations
---------------

- Projects that extends Numba should set 
  ``NUMBA_CAPTURED_ERRORS='new_style'`` for testing to find all places where 
  non-``NumbaError`` exceptions are raised during compilation.
- Modify any code that raises a non-``NumbaError`` to indicate a compilation
  error to raise a subclass of ``NumbaError`` instead. For example, instead of
  raising a ``TypeError``, raise a ``numba.core.errors.NumbaTypeError``.


Schedule
--------

- In Numba 0.58: ``NUMBA_CAPTURED_ERRORS=old_style`` is deprecated. Warnings 
  will be raised when `old_style` error capturing is used.
- In Numba 0.59: explicitly setting ``NUMBA_CAPTURED_ERRORS=old_style`` will 
  raise deprecation warnings.
- In Numba 0.60: ``NUMBA_CAPTURED_ERRORS=new_style`` becomes the default.
- In Numba 0.61: support for ``NUMBA_CAPTURED_ERRORS=old_style`` will be 
  removed.

