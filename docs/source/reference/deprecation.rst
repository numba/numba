===================
Deprecation Notices
===================

This section contains information about deprecation of behaviours, features and
APIs that have become undesirable/obsolete. Any information about the schedule
for their deprecation and reasoning behind the changes, along with examples, is
provided. However, first is a small section on how to suppress deprecation
warnings that may be raised from Numba so as to prevent warnings propagating
into code that is consuming Numba.

Suppressing Deprecation warnings
================================
All Numba deprecations are issued via ``NumbaDeprecationWarning`` or
``NumbaPendingDeprecationWarning`` s, to suppress the reporting of
these the following code snippet can be used::

    from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
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
formats at call time and and then back to CPython formats at return time). As a
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

is likely to require adjustment to use a ``typed.List`` instance (not
implemented yet!), synonymous with the ``typed.Dict``.

Schedule
--------
This feature will be removed with respect to this schedule:

* Pending-deprecation warnings will be issued in version 0.44.0
* Deprecation warnings and replacements will be issued in version 0.45.0
* Support will be removed in version 0.46.0

Recommendations
---------------
Projects that need/rely on the deprecated behaviour should pin their dependency
on Numba to a version prior to removal of this behaviour, or consider following
replacement instructions that will be issued outlining how to adjust to the
change.

Expected Replacement
--------------------
As alluded to above, it is anticipated that a ``typed.List`` (not implemented
yet!) will be used to permit similar functionality to reflection. The advantages
to this approach are:

* That the containers are typed means type inference has to work less hard.
* Nested containers (containers of containers of ...) are more easily
  supported.
* Performance penalties currently incurred translating data to/from native
  formats are largely avoided.
* Numba's ``typed.Dict`` will be able to use these containers as values.

It is expected something similar to the following will act as the replacement::

  from numba import njit, int64
  from numba.typed import List

  @njit
  def foo(x):
    x.append(10)

  a = List.empty(int64)
  a.extend([1, 2, 3])
  foo(a)


Deprecation of :term:`object mode` `fall-back` behaviour when using ``@jit``
============================================================================
The ``numba.jit`` decorator has for a long time followed the behaviour of first
attempting to compile the decorated function in :term:`nopython mode` and should
this compilation fail it will `fall-back` and try again to compile but this time
in :term:`object mode`. It it this `fall-back` behaviour which is being
deprecated, the result of which will be that ``numba.jit`` will by default
compile in :term:`nopython mode` and :term:`object mode` compilation will
become `opt-in` only.


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

Schedule
--------
This feature will be removed with respect to this schedule:

* Deprecation warnings will be issued in version 0.44.0
* Support will be removed in version 0.47.0

Recommendations
---------------
Projects that need/rely on the deprecated behaviour should pin their dependency
on Numba to a version prior to removal of this behaviour. Alternatively, to
accommodate the scheduled deprecations, users with code compiled at present with
``@jit`` can supply the ``nopython=True`` keyword argument, if the code
continues to compile then the code is already ready for this change. If the code
does not compile, continue using the ``@jit`` decorator without
``nopython=True`` and profile the performance of the function. Then remove the
decorator and again check the performance of the function. If there is no
benefit to having the ``@jit`` decorator present consider removing it! If there
is benefit to having the ``@jit`` decorator present, then to be future proof
supply the keyword argument ``forceobj=True`` to ensure the function is always
compiled in :term:`object mode`.


Deprecation of ``SmartArray``
============================================================================
``SmartArray`` is an experimental interface that provides an Array-like data
type that was intended to automatically manage data movement to and from a CUDA
compute device (GPU). It is being deprecated with view of removal in the near
future.

Reason for deprecation
----------------------
The appetite for the use of ``SmartArray`` was not large, and with advances in
both associated software and hardware any advantages it may have lead to have
been eroded away.

Example(s) of the impact
------------------------
Code using ``SmartArray`` will cease to work, e.g. this will fail::

    from numba import SmartArray
    from numba.cuda.kernels.transpose import transpose
    import numpy as np

    a = SmartArray(np.arange(16, dtype=float).reshape(4,4))
    c = SmartArray(where='gpu', shape=(4,4), dtype=float)

    transpose(a, b)

Schedule
--------
This feature will be removed with respect to this schedule:

* Deprecation warnings will be issued in version 0.44.0
* Support will be removed in version 0.45.0

Recommendations
---------------
Projects that need/rely on the deprecated behaviour should pin their dependency
on Numba to a version prior to removal of this feature and consider migrating to
using the ``numba.cuda`` API.


Deprecation of ``NUMBAPRO`` environment variables
=================================================
As a piece of legacy behaviour inherited from the time of ``NumbaPro`` it is
possible to specify various CUDA drivers, libraries and configuration to use
with Numba via the following environment variables:

* ``NUMBAPRO_NVVM``
* ``NUMBAPRO_CUDALIB``
* ``NUMBAPRO_LIBDEVICE``
* ``NUMBAPRO_CUDA_LOG_SIZE``
* ``NUMBAPRO_VERBOSE_CU_JIT_LOG``

These are being deprecated with view of removal in the near future.
Please refer to :ref:`this guide <cudatoolkit-lookup>` for alternatives to
``NUMBAPRO_NVVM``, ``NUMBAPRO_CUDALIB`` and ``NUMBAPRO_LIBDEVICE``.
For ``NUMBAPRO_CUDA_LOG_SIZE`` and ``NUMBAPRO_VERBOSE_CU_JIT_LOG``, replace
the ``NUMBAPRO`` prefix with ``NUMBA``.


Reason for deprecation
----------------------
First, ``NUMBAPRO`` as a name causes confusion for users, especially as the
product ``NumbaPro`` was retired since its technology was predominantly made
Open Source, with Numba inheriting a lot of this. Second, the environment
variables are somewhat confusing and in some cases setting one will influence
the impact of another, none of which is documented.

Schedule
--------
This feature will be removed with respect to this schedule:

* Deprecation warnings will be issued in version 0.44.0
* Support will be removed in version 0.45.0

Recommendations
---------------
Projects that need/rely on the deprecated behaviour should pin their dependency
on Numba to a version prior to removal of this behaviour, or consider following
replacement instructions outlined in :ref:`this guide <cudatoolkit-lookup>`.


Deprecation of ``numba.autojit``
================================
The decorator ``numba.autojit`` was inherited from the time of ``NumbaPro`` and
has not been recommended for use for some time.

Reason for deprecation
----------------------
It is functionally no different to ``numba.jit`` and in fact just calls out to
this function. It has not been in active use or recommended for use for a
significant period of time.

Schedule
--------
This feature will be removed with respect to this schedule:

* Deprecation warnings will be issued in version 0.44.0
* Support will be removed in version 0.47.0

Recommendations
---------------
Projects that need/rely on the deprecated behaviour should pin their dependency
on Numba to a version prior to removal of this behaviour. The recommended method
for accommodating the deprecation of ``numba.autojit`` is to simply replace it
with the semantically and functionally equivalent ``numba.jit`` decorator.


Deprecation of ``iternext_impl`` without a supplied ``RefType``
===============================================================
Whilst ``numba.targets.imputils.iternext_impl`` is technically an internal API
it has been used in external code bases as a convenient piece of functionality
when implementing the low level details of iteration.

Reason for deprecation
----------------------
The original definition of ``iternext_impl`` assumed that the reference type of
the yielded item was always ``borrowed``. The addition of ``unicode`` type
support to Numba meant that an iterator may yield an item which is a new
reference. As a result, support for ``unicode`` type iteration required a
behavioural change in ``iternext_impl`` and this was encapsulated in an API
change.

Example(s) of the impact
------------------------

In future code such as::

  from numba.targets.imputils import iternext_impl

  @iternext_impl
  def mytype_iter(context, builder, sig, args, result):
    #details

will require adjustment to use the new API.

Schedule
--------
The deprecated API will be removed with respect to this schedule:

* Deprecation warnings will be issued in version 0.44.0
* Support will be removed in version 0.46.0

Recommendations
---------------
Projects that need/rely on the deprecated behaviour should pin their dependency
on Numba to a version prior to removal of this behaviour. Alternatively, to
accommodate these changes, a direct replacement for the deprecated API is as
follows::

  from numba.targets.imputils import iternext_impl, RefType

  @iternext_impl(RefType.BORROWED)
  #              ^--- only change required is the specification of a RefType
  def mytype_iter(context, builder, sig, args, result):
    #details
