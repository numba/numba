*************************
Supported Python Features
*************************

Flow Control
============
Numba should handle Python for, while, and if/else statements in nopython mode.

Operators
=========
Most Python operators should be supported in nopython mode including:
+
-
*
/
%
**
<<
>>
&
|
^
~
-

Builtin Functions
=================
Numba currently only handles a few built in functions in nopython mode, and falls
back to object mode for the rest.

Extension Types (Classes)
=========================
Future versions of Numba will support compilation of user defined Python classes.

Containers
==========
Future versions of Numba will support Python containers including lists, tuples,
sets, and dicts in nopython mode. Numba will currently handle containers as
Python objects.

