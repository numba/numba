"""Q4 minimal reproducer — pure llvmlite, no Numba.

Trigger: two `add_ir_module` calls that produce identical post-rename strong
externals.  Crashes inside the second call (SIGSEGV) instead of returning a
DuplicateDefinition error.

What this isolates: the Q3-minimal-plan codepath (every IR submit goes
through `compileAndAdd` → `JIT->addObjectFile`). For strong-linkage
duplicates the underlying ORC `addObjectFile` does not degrade gracefully.

Run:
    python q4_repro_min.py
Expected (broken): Fatal Python error: Segmentation fault, last frame
    File "/workspace/llvmlite/binding/neworcjit.py", line 62 in add_ir_module
"""
import faulthandler; faulthandler.enable()
from llvmlite import binding as ll
ll.initialize_native_target()
ll.initialize_native_asmprinter()
from llvmlite.binding.neworcjit import create_new_orcjit

ir = r"""target triple = "aarch64-unknown-linux-gnu"
define i32 @foo(i32 %x) { ret i32 %x }
"""

eng = create_new_orcjit()
h1 = eng.add_ir_module(ir, module_id="m1")
print("h1:", h1, dict(eng._get_rename_map(h1)))
# Same IR -> same content hash -> same renamed strong external "foo.<hash>".
# Second compileAndAdd → addObjectFile encounters a duplicate strong def.
h2 = eng.add_ir_module(ir, module_id="m2")
print("h2:", h2)
print("DONE")
