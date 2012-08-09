A More Complicated Function
====================

.. code-block:: python

   #!/usr/bin/env python

   from llvm.core import *

   #create a module
   module = Module.new("tut2")

   #create a function type taking 2 integers, return a 32-bit integer
   ty_int = Type.int(32)
   func_type = Type.function(ty_int, (ty_int, ty_int))

   #create a function of that type
   gcd = Function.new(module, func_type, "gcd")

   #name function args
   x = gcd.args[0]; x.name = "x"
   y = gcd.args[1]; y.name = "y"

   #implement the function

   #blocks...
   entry = gcd.append_basic_block("entry")
   ret = gcd.append_basic_block("return") 
   cond_false = gcd.append_basic_block("cond_false")
   cond_true = gcd.append_basic_block("cond_true")
   cond_false_2 = gcd.append_basic_block("cond_false_2")

   #create a llvm::IRBuilder
   bldr = Builder.new(entry)
   x_eq_y = bldr.icmp(IPRED_EQ, x, y, "tmp")
   bldr.cbranch(x_eq_y, ret, cond_false)

   bldr.position_at_end (ret)
   bldr.ret(x)

   bldr.position_at_end(cond_false)
   x_lt_y = bldr.icmp(IPRED_ULT, x, y, "tmp")
   bldr.cbranch(x_lt_y, cond_true, cond_false_2)

   bldr.position_at_end(cond_true)
   y_sub_x = bldr.sub(y, x, "tmp")
   recur_1 = bldr.call(gcd, (x, y_sub_x,), "tmp")
   bldr.ret(recur_1)

   bldr.position_at_end(cond_false_2)
   x_sub_y = bldr.sub(x, y, "x_sub_y")
   recur_2 = bldr.call(gcd, (x_sub_y, y,), "tmp")
   bldr.ret(recur_2)

   print module
