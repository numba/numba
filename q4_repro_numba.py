"""Q4 reproducer — Numba forceobj path.

Single compile of a forceobj function whose body triggers looplift fallback.
Looplift drives a second `_finalize_final_module` of the *same* IR module
content (the lifted-loop pass re-emits the wrapper). The second
`add_ir_module` SIGSEGVs inside `JIT->addObjectFile` because the strong
externals `_ZN8__main__1fB...8pyobject8pyobject` and
`_ZN7cpython8__main__1fB...8pyobject8pyobject` were already published on
the first finalize.

Both `test_usecases.TestUsecases.test_sum1d_pyobj` and
`test_usecases.TestUsecases.test_string_conversion` hit this — they share
the forceobj-with-loop-fallback shape (sum1d has an explicit loop;
string_conversion has the `object()` builtin which forces objmode but the
return path goes through the same `_py_lowering_stage`).

Crash trace:
  add_ir_module
  _finalize_final_module (codegen.py:894)
  finalize (codegen.py:854)
  get_executable (cpu.py:284)
  _py_lowering_stage (object_mode_passes.py:89)
  ...
  _compile_ir (compiler.py:512)         <-- looplift recompile
  ...
  _compile_bytecode (compiler.py:505)   <-- original compile
"""
import faulthandler; faulthandler.enable()
from numba import jit
from numba.core import types

def f(s, e):
    c = 0
    for i in range(s, e):
        c += i
    return c

cfunc = jit((types.int32, types.int32), forceobj=True)(f)
print(cfunc(0, 5))
