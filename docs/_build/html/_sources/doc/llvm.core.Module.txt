+-----------------------------+
| layout: page                |
+-----------------------------+
| title: Module (llvm.core)   |
+-----------------------------+

Modules are top-level container objects. You need to create a module
object first, before you can add global variables, aliases or functions.
Modules are created using the static method ``Module.new``:

{% highlight python %} #!/usr/bin/env python

from llvm import \* from llvm.core import \*

create a module
===============

my\_module = Module.new('my\_module') {% endhighlight %}

The constructor of the Module class should *not* be used to instantiate
a Module object. This is a common feature for all llvmpy classes.

    **Convention**

    *All* llvmpy objects are instantiated using static methods of
    corresponding classes. Constructors *should not* be used.

    The argument ``my_module`` is a module identifier (a plain string).
    A module can also be constructed via deserialization from a bit code
    file, using the static method ``from_bitcode``. This method takes a
    file-like object as argument, i.e., it should have a ``read()``
    method that returns the entire data in a single call, as is the case
    with the builtin file object. Here is an example:

{% highlight python %} # create a module from a bit code file bcfile =
file("test.bc") my\_module = Module.from\_bitcode(bcfile) {%
endhighlight %}

There is corresponding serialization method also, called ``to_bitcode``:

{% highlight python %} # write out a bit code file from the module
bcfile = file("test.bc", "w") my\_module.to\_bitcode(bcfile) {%
endhighlight %}

Modules can also be constructed from LLVM assembly files (``.ll``
files). The static method ``from_assembly`` can be used for this.
Similar to the ``from_bitcode`` method, this one also takes a file-like
object as argument:

{% highlight python %} # create a module from an assembly file llfile =
file("test.ll") my\_module = Module.from\_assembly(llfile) {%
endhighlight %}

Modules can be converted into their assembly representation by
stringifying them (see below).

--------------

llvm.core.Module
================

-  This will become a table of contents (this text will be scraped).
   {:toc}

Static Constructors
-------------------

``new(module_id)``
~~~~~~~~~~~~~~~~~~

Create a new ``Module`` instance with given ``module_id``. The
``module_id`` should be a string.

``from_bitcode(fileobj)``
~~~~~~~~~~~~~~~~~~~~~~~~~

Create a new ``Module`` instance by deserializing the bitcode file
represented by the file-like object ``fileobj``.

``from_assembly(fileobj)``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a new ``Module`` instance by parsing the LLVM assembly file
represented by the file-like object ``fileobj``.

Properties
----------

``data_layout``
~~~~~~~~~~~~~~~

A string representing the ABI of the platform.

``target``
~~~~~~~~~~

A string like ``i386-pc-linux-gnu`` or ``i386-pc-solaris2.8``.

``pointer_size``
~~~~~~~~~~~~~~~~

[read-only]

The size in bits of pointers, of the target platform. A value of zero
represents ``llvm::Module::AnyPointerSize``.

``global_variables``
~~~~~~~~~~~~~~~~~~~~

[read-only]

An iterable that yields
`GlobalVariable <llvm.core.GlobalVariable.html>`_ objects, that
represent the global variables of the module.

``functions``
~~~~~~~~~~~~~

[read-only]

An iterable that yields `Function <llvm.core.Function.html>`_ objects,
that represent functions in the module.

``id``
~~~~~~

A string that represents the module identifier (name).

Methods
-------

``get_type_named(name)``
~~~~~~~~~~~~~~~~~~~~~~~~

Return a `StructType <llvm.core.StructType.html>`_ object for the given
name.

The definition of this method was changed to work with LLVM 3.0+, in
which the type system was rewritten. See `LLVM
Blog <http://blog.llvm.org/2011/11/llvm-30-type-system-rewrite.html>`_.

{% comment %} ++++++++REMOVED+++++++++++ ### ``add_type_name(name, ty)``

Add an alias (typedef) for the type ``ty`` with the name ``name``.

``delete_type_name(name)``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Delete an alias with the name ``name``. ++++++++END-REMOVED+++++++++++
{% endcomment %}

``add_global_variable(ty, name)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add a global variable of the type ``ty`` with the name ``name``. Returns
a `GlobalVariable <llvm.core.GlobalVariable.html>`_ object.

``get_global_variable_named(name)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get a `GlobalVariable <llvm.core.GlobalVariable.html>`_ object
corresponding to the global variable with the name ``name``. Raises
``LLVMException`` if such a variable does not exist.

``add_library(name)``
~~~~~~~~~~~~~~~~~~~~~

Add a dependent library to the Module. This only adds a name to a list
of dependent library. **No linking is performed**.

``add_function(ty, name)``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Add a function named ``name`` with the function type ``ty``. ``ty`` must
of an object of type `FunctionType <llvm.core.FunctionType.html>`_.

``get_function_named(name)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get a `Function <llvm.core.Function.html>`_ object corresponding to the
function with the name ``name``. Raises ``LLVMException`` if such a
function does not exist.

``get_or_insert_function(ty, name)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Like ``get_function_named``, but adds the function first, if not present
(like ``add_function``).

``verify()``
~~~~~~~~~~~~

Verify the correctness of the module. Raises ``LLVMException`` on
errors.

``to_bitcode(fileobj)``
~~~~~~~~~~~~~~~~~~~~~~~

Write the bitcode representation of the module to the file-like object
``fileobj``.

``link_in(other)``
~~~~~~~~~~~~~~~~~~

Link in another module ``other`` into this module. Global variables,
functions etc. are matched and resolved. The ``other`` module is no
longer valid and should not be used after this operation. This API might
be replaced with a full-fledged Linker class in the future.

Special Methods
---------------

``__str__``
~~~~~~~~~~~

``Module`` objects can be stringified into it's LLVM assembly language
representation.

``__eq__``
~~~~~~~~~~

``Module`` objects can be compared for equality. Internally, this
converts both arguments into their LLVM assembly representations and
compares the resultant strings.

    **Convention**

    *All* llvmpy objects (where it makes sense), when stringified,
    return the LLVM assembly representation. ``print module_obj`` for
    example, prints the LLVM assembly form of the entire module.

    Such objects, when compared for equality, internally compare these
    string representations.
