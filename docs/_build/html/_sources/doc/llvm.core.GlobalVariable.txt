+-------------------------------------+
| layout: page                        |
+-------------------------------------+
| title: GlobalVariable (llvm.core)   |
+-------------------------------------+

Global variables (``llvm.core.GlobalVariable``) are subclasses of
`llvm.core.GlobalValue <llvm.core.GlobalValue.html>`_ and represent
module-level variables. These can have optional initializers and can be
marked as constants. Global variables can be created either by using the
``add_global_variable`` method of the `Module <llvm.core.Module.html>`_
class, or by using the static method ``GlobalVariable.new``.

{% highlight python %} # create a global variable using
add\_global\_variable method gv1 =
module\_obj.add\_global\_variable(Type.int(), "gv1")

or equivalently, using a static constructor method
==================================================

gv2 = GlobalVariable.new(module\_obj, Type.int(), "gv2") {% endhighlight
%}

Existing global variables of a module can be accessed by name using
``module_obj.get_global_variable_named(name)`` or
``GlobalVariable.get``. All existing global variables can be enumerated
via iterating over the property ``module_obj.global_variables``.

{% highlight python %} # retrieve a reference to the global variable
gv1, # using the get\_global\_variable\_named method gv1 =
module\_obj.get\_global\_variable\_named("gv1")

or equivalently, using the static ``get`` method:
=================================================

gv2 = GlobalVariable.get(module\_obj, "gv2")

list all global variables in a module
=====================================

for gv in module\_obj.global\_variables: print gv.name, "of type",
gv.type {% endhighlight %}

The initializer for a global variable can be set by assigning to the
``initializer`` property of the object. The ``is_global_constant``
property can be used to indicate that the variable is a global constant.

Global variables can be delete using the ``delete`` method. Do not use
the object after calling ``delete`` on it.

{% highlight python %} # add an initializer 10 (32-bit integer)
gv.initializer = Constant.int( Type.int(), 10 )

delete the global
=================

gv.delete() # DO NOT dereference \`gv' beyond this point! gv = None {%
endhighlight %}

llvm.core.GlobalVariable
========================

Base Class
----------

-  `llvm.core.GlobalValue <llvm.core.GlobalValue.html>`_

Static Constructors
-------------------

``new(module_obj, ty, name)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a global variable named ``name`` of type ``ty`` in the module
``module_obj`` and return a ``GlobalVariable`` object that represents
it.

``get(module_obj, name)``
~~~~~~~~~~~~~~~~~~~~~~~~~

Return a ``GlobalVariable`` object to represent the global variable
named ``name`` in the module ``module_obj`` or raise ``LLVMException``
if such a variable does not exist.

Properties
----------

``initializer``
~~~~~~~~~~~~~~~

The intializer of the variable. Set to
`llvm.core.Constant <llvm.core.Constant.html>`_ (or derived). Gets the
initializer constant, or ``None`` if none exists. ``global_constant``
``True`` if the variable is a global constant, ``False`` otherwise.

Methods
-------

``delete()``
~~~~~~~~~~~~

Deletes the global variable from it's module. **Do not hold any
references to this object after calling ``delete`` on it.**
