+----------------------------+
| layout: page               |
+----------------------------+
| title: Value (llvm.core)   |
+----------------------------+

llvm.core.Value
===============

-  This will become a table of contents (this text will be scraped).
   {:toc}

Properties
----------

``name``
~~~~~~~~

The name of the value.

``type``
~~~~~~~~

[read-only]

An ``llvm.core.Type`` object representing the type of the value.

``uses``
~~~~~~~~

[read-only]

The list of values (``llvm.core.Value``) that use this value.

``use_count``
~~~~~~~~~~~~~

[read-only]

The number of values that use (refer) this value. Same as
``len(val.uses)`` but faster if you just want the count.

``value_id``
~~~~~~~~~~~~

[read-only]

Returns ``llvmValuegetValueID()``. Refer LLVM documentation for more
info.

Special Methods
---------------

``__str__``
~~~~~~~~~~~

``Value`` objects can be stringified into it's LLVM assembly language
representation.

``__eq__``
~~~~~~~~~~

``Value`` objects can be compared for equality. Internally, this
converts both arguments into their LLVM assembly representations and
compares the resultant strings.
