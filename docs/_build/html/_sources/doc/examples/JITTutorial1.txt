A First Function
==================

.. code-block:: python

  #!/usr/bin/env python
 
  from llvm.core import *

  #create a module
  module = Module.new("tut1")

  #create a function type taking 3 32-bit integers, return a 32-bit integer
  ty_int = Type.int(32)
  func_type = Type.function(ty_int, (ty_int,)*3)

  #create a function of that type
  mul_add = Function.new (module, func_type, "mul_add")
  mul_add.calling_convention = CC_C
  x = mul_add.args[0]; x.name = "x"
  y = mul_add.args[1]; y.name = "y"
  z = mul_add.args[2]; z.name = "z"

  #implement the function

  #new block
  blk = mul_add.append_basic_block("entry")

  #IR builder
  bldr = Builder.new(blk)
  tmp_1 = bldr.mul(x, y, "tmp_1") 
  tmp_2 = bldr.add(tmp_1, z, "tmp_2")

  bldr.ret(tmp_2)

  print module 
