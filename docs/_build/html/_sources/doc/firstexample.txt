Examples
========

A Simple Function
-----------------

Let's create a (LLVM) module containing a single function, corresponding
to the ``C`` function:

.. code-block:: c

   int sum(int a, int b)
   {
       return a + b;
   }

Here's how it looks in llvmpy:

.. code-block:: python

   #!/usr/bin/env python

   # Import the llvmpy modules.
   from llvm import *
   from llvm.core import *

   # Create an (empty) module.
   my_module = Module.new('my_module')

   # All the types involved here are "int"s. This type is represented
   # by an object of the llvm.core.Type class:
   ty_int = Type.int()     # by default 32 bits

   # We need to represent the class of functions that accept two integers
   # and return an integer. This is represented by an object of the
   # function type (llvm.core.FunctionType):
   ty_func = Type.function(ty_int, [ty_int, ty_int])

   # Now we need a function named 'sum' of this type. Functions are not
   # free-standing (in llvmpy); it needs to be contained in a module.

   f_sum = my_module.add_function(ty_func, "sum")

   # Let's name the function arguments as 'a' and 'b'.
   f_sum.args[0].name = "a" 
   f_sum.args[1].name = "b"

   # Our function needs a "basic block" -- a set of instructions that
   # end with a terminator (like return, branch etc.). By convention
   # the first block is called "entry".
   bb = f_sum.append_basic_block("entry")

   # Let's add instructions into the block. For this, we need an
   # instruction builder:
   builder = Builder.new(bb)

   # OK, now for the instructions themselves. We'll create an add
   # instruction that returns the sum as a value, which we'll use
   # a ret instruction to return.
   tmp = builder.add(f_sum.args[0], f_sum.args[1], "tmp")
   builder.ret(tmp)

   # We've completed the definition now! Let's see the LLVM assembly
   # language representation of what we've created:

   print my_module

Here is the output:

.. code-block:: llvm

   ; ModuleID = 'my_module'

   define i32 @sum(i32 %a, i32 %b) { 
   entry: 
           %tmp = add i32 %a, %b      ; <i32> [#uses=1]
           ret i32 %tmp
   }

Adding JIT Compilation
----------------------

Let's compile this function in-memory and run it.

.. code-block:: python

   #!/usr/bin/env python
  
   # Import the llvmpy modules.

   from llvm import *
   from llvm.core import * 
   from llvm.ee import *                     # new import: ee = Execution Engine

   #Create a module, as in the previous example.
   my_module = Module.new('my_module')
   ty_int = Type.int()     # by default 32 bits
   ty_func = Type.function(ty_int, [ty_int, ty_int])
   f_sum = my_module.add_function(ty_func, "sum")
   f_sum.args[0].name = "a"
   f_sum.args[1].name = "b"
   bb = f_sum.append_basic_block("entry")
   builder = Builder.new(bb)
   tmp = builder.add(f_sum.args[0], f_sum.args[1], "tmp")
   builder.ret(tmp)

   # Create an execution engine object. This will create a JIT compiler
   # on platforms that support it, or an interpreter otherwise.
   ee = ExecutionEngine.new(my_module)

   # The arguments needs to be passed as "GenericValue" objects.
   arg1 = GenericValue.int(ty_int, 100)
   arg2 = GenericValue.int(ty_int, 42)

   # Now let's compile and run!
   retval = ee.run_function(f_sum, [arg1, arg2])

   # The return value is also GenericValue. Let's print it.
   print "returned", retval.as_int()

And here's the output:

::

    returned 142
