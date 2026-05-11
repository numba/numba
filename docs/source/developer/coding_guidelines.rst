Coding Guidelines
=================

This document helps contributors match the expectations of core developers.
It covers coding conventions and APIs the project is incrementally moving
away from. Do not add new uses of discouraged APIs; replace them when
touching nearby code.


Discouraged APIs
----------------

The following APIs are being phased out. Wholesale removal is too
disruptive, so the goal is to stop adding new uses and migrate existing
ones over time.


``@lower*``
'''''''''''

Avoid the low-level ``lower_builtin``, ``lower_getattr``, ``lower_setattr``,
``lower_cast``, and ``lower_constant`` decorators from
``numba/core/imputils.py``. Prefer the high-level extension API instead:
``@overload``, ``@overload_method``, or ``@overload_attribute`` from
``numba.core.extending``.


``compile_internal``
''''''''''''''''''''

Avoid ``BaseContext.compile_internal`` from ``numba/core/base.py``.
Use ``resolve_value_type``, ``get_call_type``, and ``get_function`` instead:

.. code:: python

    disp_type = context.typing_context.resolve_value_type(func)
    sig = disp_type.get_call_type(context.typing_context, arg_types, {})
    call = context.get_function(disp_type, sig)
    result = call(builder, args)
