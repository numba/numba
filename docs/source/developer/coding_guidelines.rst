Coding Guidelines
=================

This document helps contributors match the expectations of code reviewers.
It covers coding style and conventions.


Style
-----

- Python code:

    - follows :pep:`8`
    - use `Flake8 <http://flake8.pycqa.org/en/latest/>`_. Run ``flake8 numba``.

- C code doesn't have a well-defined coding style (:pep:`7` would be nice).
- Limit to 80 column for maximum readability with all existing tools
  (such as code review UIs).


Typing annotations
------------------

- Numba uses `Mypy <http://mypy-lang.org/>`_ to verify typing annotations in
  the CI.

    - Only a subset of files are being tested (See ``mypy.ini``).
    - Only in exceptional circumstances should ``type: ignore`` comments be
      used.

- Most of Numba code base do not use type hints. We welcome incremental PRs for
  gradually adding type hints.
- New features: use type hints even if the file isn't in the mypy checklist.
- To add a file to the checklist: add it to ``files`` in ``mypy.ini`` and choose
  a compliance level (3=basic static checks, 2/1=stricter; see ``mypy.ini``
  for details).

Python's ``typing`` vs ``numba.core.typing``
''''''''''''''''''''''''''''''''''''''''''''

- ``typing`` name clash: Numba has its own ``typing`` module and
  types (e.g. ``Dict``, ``Literal``) that conflict with Python's ``typing``.
- Two options:

    - prefix Python ``typing`` imports with ``pt`` (e.g.
      ``from typing import Dict as ptDict``).
    - or ``import typing as pt``.

.. _type_anno_check:

Runtime type checking
'''''''''''''''''''''

Most code is unannotated, so runtime type checking complements mypy.

- The test suite uses `typeguard`_ to validate type annotations at runtime.
- To enable: use `runtests.py`_ as the test runner with
  ``NUMBA_USE_TYPEGUARD=1``::

    $ NUMBA_USE_TYPEGUARD=1 python runtests.py numba.tests


Discouraged APIs
----------------

The following APIs are being phased out. Wholesale removal is too
disruptive, so the goal is to stop adding new uses and migrate existing
ones over time.


Avoid ``@lower*``
'''''''''''''''''

Avoid the low-level ``lower_builtin``, ``lower_getattr``, ``lower_setattr``,
``lower_cast``, and ``lower_constant`` decorators from
``numba/core/imputils.py``. Prefer the high-level extension API instead:
``@overload``, ``@overload_method``, or ``@overload_attribute`` from
``numba.core.extending``.


Avoid ``compile_internal``
''''''''''''''''''''''''''

Avoid ``BaseContext.compile_internal`` from ``numba/core/base.py``.
Use ``resolve_value_type``, ``get_call_type``, and ``get_function`` instead:

.. code:: python

    disp_type = context.typing_context.resolve_value_type(func)
    sig = disp_type.get_call_type(context.typing_context, arg_types, {})
    call = context.get_function(disp_type, sig)
    result = call(builder, args)


.. _typeguard: https://typeguard.readthedocs.io/en/latest/
.. _runtests.py: https://github.com/numba/numba/blob/main/runtests.py
