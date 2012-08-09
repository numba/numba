+---------------------------------+
| layout: page                    |
+---------------------------------+
| title: GenericValue (llvm.ee)   |
+---------------------------------+

llvm.ee.GenericValue
====================

-  This will become a table of contents (this text will be scraped).
   {:toc}

Methods
-------

``as_int(self)``
~~~~~~~~~~~~~~~~

Return the value of this ``GenericValue`` instance as an unsigned
integer

``as_int_signed(self)``
~~~~~~~~~~~~~~~~~~~~~~~

Return the value of this ``GenericValue`` instance as a signed integer.

``as_pointer(self)``
~~~~~~~~~~~~~~~~~~~~

Return the value of this ``GenericValue`` instance as a pointer. The
type of the return value is ``int``.

``as_real(self, ty)``
~~~~~~~~~~~~~~~~~~~~~

Return the value of this ``GenericValue`` instance as a real number
which type is specified by ``ty``. ``ty`` must be a
`Type <llvm.core.Type.html>`_ instance of a real number type.

Static Factory Methods
----------------------

``int(ty, intval)``
~~~~~~~~~~~~~~~~~~~

Create a ``GenericValue`` instance with a ``int`` value, which is
zero-extended if necessary. The type of the value is specified by
``ty``, which is a `Type <llvm.core.Type.html>`_ instance.

``int_signed(ty, intval)``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a ``GenericValue`` instance with a ``int`` value, which is
sign-extended if necessary. The type of the value is specified by
``ty``, which is a `Type <llvm.core.Type.html>`_ instance.

``pointer(ty, addr)`` or ``pointer(addr)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a ``GenericValue`` instance with a ``int`` value, which is
representing a pointer value.

The two argument version is **deprecated**. The old code never used
``ty`` anyway.
