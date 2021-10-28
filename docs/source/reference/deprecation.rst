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
* Prominent notice will be given for a minimum of two releases prior to full
  removal.

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


Deprecation of the target kwarg
===============================
There have been a number of users attempting to use the ``target`` keyword
argument that's meant for internal use only. We are deprecating this argument,
as alternative solutions are available to achieve the same behaviour.

Recommendations
---------------
Update the ``jit`` decorator as follows:

* Change ``@numba.jit(..., target='cuda')`` to ``numba.cuda.jit(...)``.

Schedule
--------
This feature will be moved with respect to this schedule:

* Deprecation warnings will be issued in 0.51.0.
* The target kwarg will be removed in version 0.54.0.


Removal of the role of compute capability for CUDA inspection methods
=====================================================================

The following methods of the :class:`Dispatcher
<numba.cuda.compiler.Dispatcher>` class:

- :meth:`inspect_asm <numba.cuda.compiler.Dispatcher.inspect_asm>`
- :meth:`inspect_llvm <numba.cuda.compiler.Dispatcher.inspect_llvm>`
- :meth:`inspect_sass <numba.cuda.compiler.Dispatcher.inspect_sass>`

accepted a kwarg called ``compute_capability``. This kwarg is now removed as it
was problematic - in most cases the returned values erroneously pertained to
the device in the current context, instead of the requested compute capability.

These methods return a dict of variants, which was previously keyed by a
``(compute_capability, argtypes)`` tuple. The dict is now only keyed by
argument types, and items in the dict are for the device in the current
context.

For specialized Dispatchers (those whose kernels were eagerly compiled by
providing a signature), the methods previously returned only one variant,
instead of a dict of variants. For consistency with the CPU target and for
support for multiple signatures to be added to the CUDA target, these methods
now always return a dict.

The :meth:`ptx <numba.cuda.compiler.Dispatcher.ptx>` property also returned one
variant directly for specialized Dispatchers, and a dict for un-specialized
Dispatchers. It now always returns a dict

Recommendations
---------------

Update calls to these methods such that:

- They are always called when the device for which their output is required is
  in the current CUDA context.
- The ``compute_capability`` kwarg is not passed to them.
- Any use of their results indexes into them using only a tuple of argument
  types.
- With specialized Dispatchers, ensure that the returned dict is indexed into
  using the appropriate signature.

Schedule
--------

In 0.53.0:

- The ``compute_capability`` kwarg was deprecated.
- Returned values from the inspection methods supported indexing by
  ``(compute_capability, argtypes)`` and ``argtypes``.
- The inspection methods and ``ptx`` property of specialized dispatchers returned
  their result for a single variant, rather than a dict, and produced a
  warning.

In 0.54.0:

- The ``compute_capability`` kwarg has been removed.
- ``ptx`` and the inspection methods always return a dict.
- Support for indexing into the results of these methods using ``(cc,
  argtypes)`` has been removed.


.. _deprecation-strict-strides:

Deprecation of strict strides checking when computing contiguity
================================================================

The contiguity of device arrays (the ``'C_CONTIGUOUS'`` and ``'F_CONTIGUOUS'``
elements of the flags of a device array) are computed using relaxed strides
checking, which matches the default in NumPy since Version 1.12. A config
variable, :envvar:`NUMBA_NPY_RELAXED_STRIDES_CHECKING`, is provided to force
computation of these flags using strict strides checking.

This flag is provided to work around any bugs that may be exposed by strict
strides checking, and will be removed in future.

Schedule
--------

In 0.54.0:

- Relaxed strides checking will become the default.
- Strict strides checking will be deprecated.

In 0.55.0:

- Strict strides checking will be removed, if there are no reports of bugs
  related to relaxed strides checking in 0.54.0 onwards. This plan will be
  re-examined if bugs related to relaxed strides checking are reported, but may
  not necessarily change as a result.


Deprecation of the ``inspect_ptx()`` method
===========================================

The undocumented ``inspect_ptx()`` method of functions decorated with
``@cuda.jit(device=True)`` is sometimes used to compile a Python function to
PTX for use outside of Numba. An interface for this specific purpose is
provided in the :func:`compile_ptx() <numba.cuda.compile_ptx>` function.
``inspect_ptx()`` has one or two longstanding issues and presents a maintenance
burden for upcoming changes in the CUDA target, so it is deprecated and will be
removed in favor of the use of :func:`compile_ptx() <numba.cuda.compile_ptx>`.

Recommendations
---------------

Replace any code that compiles device functions to PTX using the following
pattern:

.. code-block:: python

    @cuda.jit(signature, device=True)
    def func(args):
        ...

    ptx_code = func.inspect_ptx(nvvm_options=nvvm_options).decode()

with:

.. code-block:: python

    def func(args):
        ...

    ptx_code, return_type = compile_ptx(func, signature, device=True, nvvm_options=nvvm_options)

Schedule
--------

- In Numba 0.54: ``inspect_ptx()`` will be deprecated.
- In Numba 0.55: ``inspect_ptx()`` will be removed.


Deprecation of eager compilation of CUDA device functions
=========================================================

In future versions of Numba, the ``device`` kwarg to the ``@cuda.jit`` decorator
will be obviated, and whether a device function or global kernel is compiled will
be inferred from the context. With respect to kernel / device functions and lazy
/ eager compilation, four cases are presently handled:

1. ``device=True``, eager compilation with a signature provided
2. ``device=False``, eager compilation with a signature provided
3. ``device=True``, lazy compilation with no signature
4. ``device=False``, lazy compilation with no signature

The latter two cases can be differentiated without the ``device`` kwarg, because
it can be inferred from the calling context - if the call is from the host, then
a global kernel should be compiled, and if the call is from a kernel or another
device function, then a device function should be compiled.

The first two cases cannot be differentiated in the absence of the ``device``
kwarg - without it, it will not be clear from a signature alone whether a device
function or global kernel should be compiled. In order to resolve this, support
for eager compilation of device functions will be removed. Eager compilation
with the ``@cuda.jit`` decorator will in future always imply the immediate
compilation of a global kernel.

Recommendations
---------------

Any eagerly-compiled device functions should have their signature removed, e.g.:

.. code-block:: python

   @cuda.jit('int32(int32, int32)', device=True)
   def f(x, y):
       return x + y

becomes:


.. code-block:: python

   @cuda.jit(device=True)
   def f(x, y):
       return x + y

Schedule
--------

- In Numba 0.54: Eager compilation of device functions will be deprecated.
- In Numba 0.55: Eager compilation of device functions will be unsupported and
  attempts to eagerly compile device functions will raise an error.


.. _rocm_unmaintained:

Dropping support for the ROCm target
====================================

The `ROCm <https://rocmdocs.amd.com/en/latest/index.html>`_ target has not been
maintained for a number of years. It's known to be not far from working but has
essentially bit-rotted in a number of areas. Numba 0.54 includes a new API for
describing targets and both the CPU and CUDA targets have been ported to use
this. Due to lack of maintenance, support and user base, the ROCm target is
not being ported to this API, is being moved to an "unmaintained" status and
will reside outside of the Numba package. Should there be sufficient interest
and support for this target in future its status will be reconsidered.

Schedule
--------

In 0.54.0:

- The ``ROCm`` target is officially unmaintained and the target source code has
  been moved out of the Numba main repository and into a `separate repository
  <https://github.com/numba/numba-rocm>`_.
