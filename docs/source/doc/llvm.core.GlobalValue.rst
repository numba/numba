+----------------------------------+
| layout: page                     |
+----------------------------------+
| title: GlobalValue (llvm.core)   |
+----------------------------------+

The class ``llvm.core.GlobalValue`` represents module-scope aliases,
variables and functions. Global variables are represented by the
sub-class `llvm.core.GlobalVariable <llvm.core.GlobalVariable.html>`_
and functions by `llvm.core.Function <llvm.core.Function.html>`_.

Global values have the read-write properties ``linkage``, ``section``,
``visibility`` and ``alignment``. Use one of the following constants
(from llvm.core) as values for ``linkage`` (see `LLVM
documentaion <http://www.llvm.org/docs/LangRef.html#linkage>`_ for
details on each):

Value \| Equivalent LLVM Assembly Keyword \|
------\|----------------------------------\| ``LINKAGE_EXTERNAL`` \|
``externally_visible`` \| ``LINKAGE_AVAILABLE_EXTERNALLY`` \|
``available_externally`` \| ``LINKAGE_LINKONCE_ANY`` \| ``linkonce`` \|
``LINKAGE_LINKONCE_ODR`` \| ``linkonce_odr`` \| ``LINKAGE_WEAK_ANY`` \|
``weak`` \| ``LINKAGE_WEAK_ODR`` \| ``weak_odr`` \|
``LINKAGE_APPENDING`` \| ``appending`` \| ``LINKAGE_INTERNAL`` \|
``internal`` \| ``LINKAGE_PRIVATE`` \| ``private`` \|
``LINKAGE_DLLIMPORT`` \| ``dllimport`` \| ``LINKAGE_DLLEXPORT`` \|
``dllexport`` \| ``LINKAGE_EXTERNAL_WEAK`` \| ``extern_weak`` \|
``LINKAGE_GHOST`` \| deprecated -- do not use \| ``LINKAGE_COMMON`` \|
``common`` \| ``LINKAGE_LINKER_PRIVATE`` \| ``linker_private`` \|

The ``section`` property can be assigned strings (like ".rodata"), which
will be used if the target supports it. Visibility property can be set
to one of thse constants (from llvm.core, see also `LLVM
docs <http://www.llvm.org/docs/LangRef.html#visibility>`_):

Value \| Equivalent LLVM Assembly Keyword \|
------\|----------------------------------\| ``VISIBILITY_DEFAULT`` \|
``default`` \| ``VISIBILITY_HIDDEN`` \| ``hidden`` \|
``VISIBILITY_PROTECTED`` \| ``protected`` \|

The ``alignment`` property can be 0 (default), or can be set to a power
of 2. The read-only property ``is_declaration`` can be used to check if
the global is a declaration or not. The module to which the global
belongs to can be retrieved using the ``module`` property (read-only).

llvm.core.GlobalValue
=====================

-  This will become a table of contents (this text will be scraped).
   {:toc}

Base Class
----------

-  `llvm.core.Constant <llvm.core.Constant.html>`_

Properties
----------

``linkage``
~~~~~~~~~~~

The linkage type, takes one of the constants listed above (LINKAGE\_\*).

``section``
~~~~~~~~~~~

A string like ".rodata", indicating the section into which the global is
placed into.

``visibility``
~~~~~~~~~~~~~~

The visibility type, takes one of the constants listed above
(VISIBILITY\_\*).

``alignment``
~~~~~~~~~~~~~

A power-of-2 integer indicating the boundary to align to.

``is_declaration``
~~~~~~~~~~~~~~~~~~

[read-only]

``True`` if the global is a declaration, ``False`` otherwise.

``module``
~~~~~~~~~~

[read-only]

::

    The module object to which this global belongs to.

