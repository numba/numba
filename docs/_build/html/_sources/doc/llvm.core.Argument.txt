+-------------------------------+
| layout: page                  |
+-------------------------------+
| title: Argument (llvm.core)   |
+-------------------------------+

The ``args`` property of ``llvm.core.Function`` objects yields
``llvm.core.Argument`` objects. This allows for setting attributes for
functions arguments. ``Argument`` objects cannot be constructed from
user code, the only way to get a reference to these are from
``Function`` objects.

The method ``add_attribute`` and ``remove_attribute`` can be used to add
or remove the following attributes:

Value\| Equivalent LLVM Assembly Keyword \|
-----\|----------------------------------\| ``ATTR_ZEXT``\ \|
``zeroext`` \| ``ATTR_SEXT``\ \| ``signext`` \| ``ATTR_IN_REG``\ \|
``inreg`` \| ``ATTR_BY_VAL``\ \| ``byval`` \| ``ATTR_STRUCT_RET``\ \|
``sret`` \| ``ATTR_NO_ALIAS``\ \| ``noalias`` \| ``ATTR_NO_CAPTURE``\ \|
``nocapture`` \| ``ATTR_NEST``\ \| ``nest`` \|

These method work exactly like the `corresponding
methods <functions.html#fnattr>`_ of the ``Function`` class above. Refer
`LLVM docs <http://www.llvm.org/docs/LangRef.html#paramattrs>`_ for
information on what each attribute means.

The alignment of any argument can be set via the ``alignment`` property,
to any power of 2.

llvm.core.Argument
==================

Base Class
----------

-  `llvm.core.Value <llvm.core.Value.html>`_

Properties
----------

``alignment``
~~~~~~~~~~~~~

The alignment of the argument. Must be a power of 2.

Methods
-------

``add_attribute(attr)``
~~~~~~~~~~~~~~~~~~~~~~~

Add an attribute ``attr`` to the argument, from the set listed above.

``remove_attribute(attr)``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Remove the attribute ``attr`` of the argument.
