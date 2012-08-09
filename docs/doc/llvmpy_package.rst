***********************
The llvmpy Package 
***********************

The llvmpy is a Python package, consisting of 6 modules, that wrap over
enough LLVM APIs to allow the implementation of your own compiler/VM
backend in pure Python. If you're come this far, you probably know why
this is a good idea.

Out of the 6 modules, one is an "extension" module (i.e., it is written
in C), and another one is a small private utility module, which leaves 4
public modules. These are:

-  *llvm* -- top-level package, common classes (like exceptions)
-  *llvm.core* -- IR-related APIs
-  *llvm.ee* -- execution engine related APIs
-  *llvm.passes* -- pass manager and passes related APIs

The modules contain only classes and (integer) constants. Mostly simple
Python constructs are used (deliberately) --
`property() <http://docs.python.org/lib/built-in-funcs.html>`_ and
`property
decorators <http://wiki.python.org/moin/PythonDecoratorLibrary>`_ are
probably the most exotic animals around. All classes are "new style"
classes. The APIs are designed to be navigable (and guessable!) once you
know a few conventions. These conventions are highlighted in the
sections below.

Here is a quick overview of the contents of each package:

llvm
----

-  LLVMException -- exception class (currently the only one)

llvm.core
---------

-  `Module <llvm.core.Module.html>`_ -- represents an LLVM Module
-  `Type <types.html>`_ -- represents an LLVM Type
-  `Value <values.html>`_ -- represents an LLVM Value, including:
   globals, constants, variables, arguments, functions, instructions,
   etc..
-  `BasicBlock <llvm.core.BasicBlock.html>`_ -- another derived of
   Value, represents an LLVM basic block
-  `Builder <llvm.core.Builder.html>`_ -- used for creating
   instructions, wraps LLVM IRBuilder helper class
-  constants *TYPE\_\** that represents various types
-  constants *CC\_\** that represent calling conventions
-  constants *ICMP\_\** and *FCMP\_\** that represent integer and real
   comparison predicates (like less than, greater than etc.)
-  constants *LINKAGE\_\** that represent linkage of symbols (external,
   internal etc.)
-  constants *VISIBILITY\_\** that represents visibility of symbols
   (default, hidden, protected)
-  constants *ATTR\_\** that represent function parameter attributes

llvm.ee
-------

-  `ExecutionEngine <llvm.ee.ExecutionEngine.html>`_ -- represents an
   execution engine (which can be an either an interpreter or a JIT)
-  `TargetData <llvm.ee.TargetData.html>`_ -- represents the ABI of the
   target platform (details like sizes and alignment of primitive types,
   endinanness etc)

llvm.passes
-----------

-  `PassManager <llvm.passes.PassManager.html>`_ -- represents an LLVM
   pass manager
-  `FunctionPassManager <llvm.passes.FunctionPassManager.html>`_ --
   represents an LLVM function pass manager
-  constants *PASS\_\** that represent various passes

A note on the importing of these modules
----------------------------------------

Pythonically, modules are imported with the statement
``import llvm.core``. However, you might find it more convenient to
import llvmpy modules thus:

{% highlight python %} from llvm import \* from llvm.core import \* from
llvm.ee import \* from llvm.passes import \* {% endhighlight %}

This avoids quite some typing. Both conventions work, however.

    **Tip**

    Python-style documentation strings (``__doc__``) are present in
    llvmpy. You can use the ``help()`` of the interactive Python
    interpreter or the ``object?`` of
    `IPython <http://ipython.scipy.org/moin/>`_ to get online help.
    (Note: not complete yet!)
