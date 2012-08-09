+-------------------------------+
| layout: page                  |
+-------------------------------+
| title: TargetData (llvm.ee)   |
+-------------------------------+

llvm.ee.TargetData
==================

-  This will become a table of contents (this text will be scraped).
   {:toc}

Methods
-------

``abi_alignment(self, ty)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Returns the minimum ABI-required alignment for the specified type
``ty``.

``abi_size(self, ty)``
~~~~~~~~~~~~~~~~~~~~~~

``callframe_alignment(self, ty)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Returns the minimum ABI-required alignment for the specified type ``ty``
when it is part of a call frame.

``element_at_offset(self, ty, ofs)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``offset_of_element(self, ty, el)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``preferred_alignment(self, ty_or_gv)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``size(self, ty)``
~~~~~~~~~~~~~~~~~~

``store_size(self, ty)``
~~~~~~~~~~~~~~~~~~~~~~~~

``__str__(self)``
~~~~~~~~~~~~~~~~~

Returns the string representation.

Static Factory Methods
----------------------

``new(strrep)``
~~~~~~~~~~~~~~~

Construct a new ``TargetData`` instance from the string representation

Properties
----------

``byte_order``
~~~~~~~~~~~~~~

``pointer_size``
~~~~~~~~~~~~~~~~

``target_integer_type``
~~~~~~~~~~~~~~~~~~~~~~~

