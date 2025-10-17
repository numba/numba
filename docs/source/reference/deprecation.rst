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

The use of the ``NUMBA_CAPTURED_ERRORS`` environment variable is deprecated and
removed.

Reason for deprecation
----------------------

Previously, this variable allowed controlling how Numba handles exceptions 
during compilation that do not inherit from ``numba.core.errors.NumbaError``. 
The default "old_style" behavior was to capture and wrap these errors, often 
obscuring the original exception.

The new "new_style" option treats non-``NumbaError`` exceptions as hard errors, 
propagating them without capturing. This differentiates compilation errors from 
unintended exceptions during compilation.

The old style was removed in favor of the new behavior.

Impact
------

The impact of this deprecation will only affect those who are extending Numba
functionality. 

Recommendations
---------------

- Modify any code that raises a non-``NumbaError`` to indicate a compilation
  error to raise a subclass of ``NumbaError`` instead. For example, instead of
  raising a ``TypeError``, raise a ``numba.core.errors.NumbaTypeError``.


Schedule
--------

- In Numba 0.58: ``NUMBA_CAPTURED_ERRORS=old_style`` was deprecated. Warnings
  will be raised when `old_style` error capturing is used.
- In Numba 0.59: explicitly setting ``NUMBA_CAPTURED_ERRORS=old_style`` will 
  raise deprecation warnings.
- In Numba 0.60: ``NUMBA_CAPTURED_ERRORS=new_style`` became the default.
- In Numba 0.61: support for ``NUMBA_CAPTURED_ERRORS=old_style`` was removed.

.. _cuda-builtin-target-deprecation-notice:

Deprecation of the built-in CUDA target
=======================================

The CUDA target is now maintained in a separate package, `numba-cuda
<https://nvidia.github.io/numba-cuda>`_, and the built-in CUDA target is
deprecated.

Reason for deprecation
----------------------

Development of the CUDA target has been moved to the ``numba-cuda`` package to
proceed independently of Numba development. See :ref:`cuda-deprecation-status`.

Impact
------

The built-in CUDA target is still supported by Numba 0.61 and will continue to
be provided through at least Numba 0.62, but new changes to the built-in target
are not expected; bug fixes and new features will be added in ``numba-cuda``. No
code changes are required to any code that uses the CUDA target.

Recommendations
---------------

Users should install the ``numba-cuda`` package when using the CUDA target.

To install ``numba-cuda`` with ``pip``::

   pip install numba-cuda

To install ``numba-cuda`` with ``conda``, for example from the ``conda-forge``
channel::

   conda install conda-forge::numba-cuda


Maintainers of packages that use the CUDA target should add ``numba-cuda`` as a
dependency in addition to ``numba``, or replace the ``numba`` dependency with
``numba-cuda`` if the CUDA target is used exclusively.


Schedule
--------

- In Numba 0.61: The built-in CUDA target is deprecated.
- In Numba 0.63: Use of the CUDA target when the ``numba-cuda`` package is not
  installed will cause the emission of a warning prompting the installation of
  ``numba-cuda``.
- In a future version of Numba no less than 0.63: The built-in CUDA target will
  be removed, and use of the CUDA target in the absence of the ``numba-cuda``
  package will raise an error.
