+--------------------------------+
| layout: page                   |
+--------------------------------+
| title: Comparison Operations   |
+--------------------------------+

Integer Comparision # {#icmp}
=============================

Predicates for use with ``icmp`` instruction are listed below. All of
these are integer constants defined in the ``llvm.core`` module.

``ICMP_EQ``
-----------

Equality

``ICMP_NE``
-----------

Inequality

``ICMP_UGT``
------------

Unsigned greater than

``ICMP_UGE``
------------

Unsigned greater than or equal

``ICMP_ULT``
------------

Unsigned less than

``ICMP_ULE``
------------

Unsigned less than or equal

``ICMP_SGT``
------------

Signed greater than

``ICMP_SGE``
------------

Signed greater than or equal

``ICMP_SLT``
------------

Signed less than

``ICMP_SLE``
------------

Signed less than or equal

Float Comparision # {#fcmp}
===========================

Predicates for use with ``fcmp`` instruction are listed below. All of
these are integer constants defined in the ``llvm.core`` module.

``FCMP_FALSE``
--------------

Always false

``FCMP_OEQ``
------------

True if ordered and equal

``FCMP_OGT``
------------

True if ordered and greater than

``FCMP_OGE``
------------

True if ordered and greater than or equal

``FCMP_OLT``
------------

True if ordered and less than

``FCMP_OLE``
------------

True if ordered and less than or equal

``FCMP_ONE``
------------

True if ordered and operands are unequal

``FCMP_ORD``
------------

True if ordered (no NaNs)

``FCMP_UNO``
------------

True if unordered: ``isnan(X) | isnan(Y)``

``FCMP_UEQ``
------------

True if unordered or equal

``FCMP_UGT``
------------

True if unordered or greater than

``FCMP_UGE``
------------

True if unordered, greater than or equal

``FCMP_ULT``
------------

True if unordered, or less than

``FCMP_ULE``
------------

True if unordered, less than or equal

``FCMP_UNE``
------------

True if unordered or not equal

``FCMP_TRUE``
-------------

Always true
