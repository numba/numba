+--------------------+
| layout: page       |
+--------------------+
| title: Functions   |
+--------------------+

Functions are represented by
`llvm.core.Function <llvm.core.Function.html>`_ objects. They are
contained within modules, and can be created either with the method
``module_obj.add_function`` or the static constructor ``Function.new``.
References to functions already present in a module can be retrieved via
``module.get_function_named`` or by the static constructor method
``Function.get``. All functions in a module can be enumerated by
iterating over ``module_obj.functions``.

{% highlight python %} # create a type, representing functions that take
an integer and return # a floating point value. ft = Type.function(
Type.float(), [ Type.int() ] )

create a function of this type
==============================

f1 = module\_obj.add\_function(ft, "func1")

or equivalently, like this:
===========================

f2 = Function.new(module\_obj, ft, "func2")

get a reference to an existing function
=======================================

f3 = module\_obj.get\_function\_named("func3")

or like this:
=============

f4 = Function.get(module\_obj, "func4")

list all function names in a module
===================================

for f in module\_obj.functions: print f.name {% endhighlight %}

Intrinsic
=========

References to intrinsic functions can be got via the static constructor
``intrinsic``. This returns a ``Function`` object, calling which is
equivalent to invoking the intrinsic. The ``intrinsic`` method has to be
called with a module object, an intrinsic ID (which is a numeric
constant) and a list of the types of arguments (which LLVM uses to
resolve overloaded intrinsic functions).

{% highlight python %} # get a reference to the llvm.bswap intrinsic
bswap = Function.intrinsic(mod, INTR\_BSWAP, [Type.int()])

call it
=======

builder.call(bswap, [value]) {% endhighlight %}

Here, the constant ``INTR_BSWAP``, available from ``llvm.core``,
represents the LLVM intrinsic
`llvm.bswap <http://www.llvm.org/docs/LangRef.html#int_bswap>`_. The
``[Type.int()]`` selects the version of ``llvm.bswap`` that has a single
32-bit integer argument. The list of intrinsic IDs defined as integer
constants in ``llvm.core``. These are:

{% include intrinsics.csv %}

There are also target-specific intrinsics (which correspond to that
target's CPU instructions) available, but are omitted here for brevity.
Full list can be seen from
[*intrinsic\_ids.py](https://github.com/numba/llvmpy/blob/master/llvm/*\ intrinsic\_ids.py).
See the `LLVM Language
Reference <http://www.llvm.org/docs/LangRef.html>`_ for more information
on the intrinsics, and the
`test <https://github.com/numba/llvmpy/blob/master/test/intrinsic.py>`_
directory in the source distribution for more examples. The intrinsic ID
can be retrieved from a function object with the read-only property
``intrinsic_id``.

    **Auto-generation of Intrinsic IDs**

    A script (tool/intrgen.py in source tree) generates the intrinsic
    IDs automatically. This is necessary when compiling llvmpy with a
    different version of LLVM.

Calling Convention # {#callconv}
================================

The function's calling convention can be set using the
``calling_convention`` property. The following (integer) constants
defined in ``llvm.core`` can be used as values:

Value \| Equivalent LLVM Assembly Keyword \|
------\|----------------------------------\| ``CC_C`` \| ``ccc`` \|
``CC_FASTCALL`` \| ``fastcc`` \| ``CC_COLDCALL`` \| ``coldcc`` \|
``CC_X86_STDCALL`` \| ``x86_stdcallcc`` \| ``CC_X86_FASTCALL`` \|
``x86_fastcallcc`` \|

See the `LLVM docs <http://www.llvm.org/docs/LangRef.html#callingconv>`_
for more information on each. Backend-specific numbered conventions can
be directly passed as integers.

An arbitrary string identifying which garbage collector to use can be
set or got with the property ``collector``.

The value objects corresponding to the arguments of a function can be
got using the read-only property ``args``. These can be iterated over,
and also be indexed via integers. An example:

{% highlight python %} # list all argument names and types for arg in
fn.args: print arg.name, "of type", arg.type

change the name of the first argument
=====================================

fn.args[0].name = "objptr" {% endhighlight %}

Basic blocks (see later) are contained within functions. When newly
created, a function has no basic blocks. They have to be added
explicitly, using the ``append_basic_block`` method, which adds a new,
empty basic block as the last one in the function. The first basic block
of the function can be retrieved using the ``get_entry_basic_block``
method. The existing basic blocks can be enumerated by iterating over
using the read-only property ``basic_blocks``. The number of basic
blocks can be got via ``basic_block_count`` method. Note that
``get_entry_basic_block`` is slightly faster than ``basic_blocks[0]``
and so is ``basic_block_count``, over ``len(f.basic_blocks)``.

{% highlight python %} # add a basic block b1 =
fn.append\_basic\_block("entry")

get the first one
=================

b2 = fn.get\_entry\_basic\_block() b2 = fn.basic\_mdblocks[0] # slower
than previous method

print names of all basic blocks
===============================

for b in fn.basic\_blocks: print b.name

get number of basic blocks
==========================

n = fn.basic\_block\_count n = len(fn.basic\_blocks) # slower than
previous method {% endhighlight %}

Functions can be deleted using the method ``delete``. This deletes them
from their containing module. All references to the function object
should be dropped after ``delete`` has been called.

Functions can be verified with the ``verify`` method. Note that this may
not work properly (aborts on errors).

Function Attributes # {#fnattr}
===============================

Function attributes, as documented
`here <http://www.llvm.org/docs/LangRef.html#fnattrs>`_, can be set on
functions using the methods ``add_attribute`` and ``remove_attribute``.
The following values may be used to refer to the LLVM attributes:

Value \| Equivalent LLVM Assembly Keyword \|
------\|----------------------------------\|
``ATTR_ALWAYS_INLINE``\ \|\ ``alwaysinline`` \|
``ATTR_INLINE_HINT``\ \|\ ``inlinehint`` \|
``ATTR_NO_INLINE``\ \|\ ``noinline`` \|
``ATTR_OPTIMIZE_FOR_SIZE``\ \|\ ``optsize`` \|
``ATTR_NO_RETURN``\ \|\ ``noreturn`` \|
``ATTR_NO_UNWIND``\ \|\ ``nounwind`` \|
``ATTR_READ_NONE``\ \|\ ``readnone`` \|
``ATTR_READONLY``\ \|\ ``readonly`` \|
``ATTR_STACK_PROTECT``\ \|\ ``ssp`` \|
``ATTR_STACK_PROTECT_REQ``\ \|\ ``sspreq`` \|
``ATTR_NO_REDZONE``\ \|\ ``noredzone`` \|
``ATTR_NO_IMPLICIT_FLOAT``\ \|\ ``noimplicitfloat`` \|
``ATTR_NAKED``\ \|\ ``naked`` \|

Here is how attributes can be set and removed:

{% highlight python %} # create a function ti = Type.int(32) tf =
Type.function(ti, [ti, ti]) m = Module.new('mod') f =
m.add\_function(tf, 'sum') print f # declare i32 @sum(i32, i32)

add a couple of attributes
==========================

f.add\_attribute(ATTR\_NO\_UNWIND) f.add\_attribute(ATTR\_READONLY)
print f # declare i32 @sum(i32, i32) nounwind readonly {% endhighlight
%}

**Related Links**

`llvm.core.Function <llvm.core.Function.html>`_,
`llvm.core.Argument <llvm.core.Argument.html>`_
