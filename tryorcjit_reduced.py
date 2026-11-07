"""
Reduced repro for the ``Symbols not found: [ NRT_MemInfo_call_dtor ]``
failure that ``python tryorcjit.py`` hits on the ``feature/orcjit`` branch.

Two phases:

1. CAPTURE: monkeypatch ``codegen.JitEngine.add_module`` so we record every
   IR module Numba feeds the JIT, then trigger the failing compile of
   ``@njit def foo(x): return x + 1; foo(123)``. Modules are dumped to
   ``/tmp/numba_mod_<i>.ll``.

2. REPLAY: spin up a fresh ``llvmlite.binding`` ``NewOrcJIT``, mirror the
   ``ll.add_symbol(...)`` registrations Numba's nrt.py performs, then add
   each captured module and call ``finalize_object()``. This is the same
   sequence Numba performs, with no Numba code paths in the critical
   section.

Run::

    python /workspace/numba/tryorcjit_reduced.py

It prints either ``REPLAY FINALIZE OK`` or a clean error pointing at the
missing symbol, plus diagnostics on whether ``NRT_MemInfo_call_dtor`` was
visible in ``_nrt.c_helpers`` and ``ll.add_symbol`` at the time the JIT
looked it up.
"""

import os
import sys
import glob
import traceback


CAPTURE_DIR = "/tmp"
CAPTURE_PREFIX = "numba_mod_"


def clean_capture_dir():
    for p in glob.glob(os.path.join(CAPTURE_DIR, CAPTURE_PREFIX + "*.ll")):
        try:
            os.remove(p)
        except OSError:
            pass


def phase_capture():
    print("=" * 72)
    print("PHASE 1: capture IR modules Numba feeds the JIT")
    print("=" * 72)

    clean_capture_dir()

    # Import Numba and patch JitEngine.add_module BEFORE the first njit
    # compile triggers anything.
    from numba.core import codegen as _cg

    captured = []
    orig_add = _cg.JitEngine.add_module

    def spy(self, module):
        idx = len(captured)
        ir_text = str(module)
        captured.append(ir_text)
        path = os.path.join(CAPTURE_DIR, f"{CAPTURE_PREFIX}{idx}.ll")
        with open(path, "w") as fh:
            fh.write(ir_text)
        print(f"  [capture] module #{idx} -> {path} "
              f"({len(ir_text)} bytes)")
        return orig_add(self, module)

    _cg.JitEngine.add_module = spy

    from numba import njit

    @njit
    def foo(x):
        return x + 1

    try:
        res = foo(123)
        print(f"  Numba compile + run succeeded: foo(123) = {res}")
    except Exception as e:
        print(f"  Numba compile failed (expected on feature/orcjit): "
              f"{type(e).__name__}: {e}")

    # Diagnostics: does _nrt.c_helpers actually contain MemInfo_call_dtor?
    try:
        import numba.core.runtime._nrt_python as _nrt
        keys = list(_nrt.c_helpers.keys())
        wanted = "MemInfo_call_dtor"
        present = wanted in keys
        print(f"\n  _nrt.c_helpers has 'MemInfo_call_dtor': {present}")
        # And via the same renaming nrt.py does, would it become
        # 'NRT_MemInfo_call_dtor'?
        canonical = {("NRT_" + k) if not k.startswith("_") else k
                     for k in keys}
        print(f"  'NRT_MemInfo_call_dtor' in canonical-names: "
              f"{'NRT_MemInfo_call_dtor' in canonical}")
    except Exception as e:
        print(f"  could not inspect _nrt.c_helpers: {e}")

    # Look at which captured module(s) declare NRT_MemInfo_call_dtor.
    print("\n  Modules referencing NRT_MemInfo_call_dtor:")
    any_ref = False
    for i, ir in enumerate(captured):
        if "NRT_MemInfo_call_dtor" in ir:
            any_ref = True
            # Find first reference line.
            for line in ir.splitlines():
                if "NRT_MemInfo_call_dtor" in line:
                    print(f"    mod#{i}: {line.strip()[:120]}")
                    break
    if not any_ref:
        print("    (NONE — captured IR doesn't declare it; bug is upstream)")

    print(f"\n  captured {len(captured)} module(s) total")
    return captured


def phase_replay():
    print()
    print("=" * 72)
    print("PHASE 2: replay captured modules through a fresh NewOrcJIT")
    print("=" * 72)

    # Use a fresh subprocess-like state by importing llvmlite directly.
    # Note: llvmlite's process-globals (initialize_*, add_symbol table)
    # are already initialised by phase 1's Numba import, which is fine —
    # add_symbol calls are idempotent and we want the same global state.
    import llvmlite.binding as ll

    # Note: ll.initialize() is deprecated in this llvmlite — LLVM init is
    # automatic. Numba's import already wired everything anyway.


    # Mirror nrt.py:35-42 — register every NRT_* C helper.
    import numba.core.runtime._nrt_python as _nrt
    n_registered = 0
    saw_call_dtor = False
    for py_name, addr in _nrt.c_helpers.items():
        if py_name.startswith("_"):
            c_name = py_name
        else:
            c_name = "NRT_" + py_name
        ll.add_symbol(c_name, addr)
        n_registered += 1
        if c_name == "NRT_MemInfo_call_dtor":
            saw_call_dtor = True
            print(f"  [add_symbol] {c_name} -> 0x{addr:x}")
    print(f"  registered {n_registered} symbols via ll.add_symbol")
    print(f"  NRT_MemInfo_call_dtor registered: {saw_call_dtor}")

    # Build a fresh NewOrcJIT.
    eng = ll.create_new_orcjit()

    paths = sorted(glob.glob(os.path.join(CAPTURE_DIR,
                                          CAPTURE_PREFIX + "*.ll")),
                   key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
    if not paths:
        print("  no captured modules — phase 1 must have failed before "
              "the first add_module call")
        return False

    print(f"  replaying {len(paths)} module(s)...")
    for path in paths:
        with open(path) as fh:
            ir = fh.read()
        try:
            h = eng.add_ir_module(ir)
            print(f"    add_ir_module({os.path.basename(path)}) -> "
                  f"handle={h}")
        except Exception as e:
            print(f"    add_ir_module({os.path.basename(path)}) FAILED: "
                  f"{type(e).__name__}: {e}")
            return False

    print("  calling finalize_object() ...")
    try:
        eng.finalize_object()
        print("  REPLAY FINALIZE OK")
        return True
    except Exception as e:
        print(f"  REPLAY FINALIZE FAILED: {type(e).__name__}: {e}")
        return False


def main():
    captured = phase_capture()
    if not captured:
        print("\nNo IR was captured — nothing to replay.")
        return 2

    ok = phase_replay()

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    if ok:
        print("Replay succeeded. The reduced path through llvmlite works.")
        print("=> Numba's real-life path differs from the captured IR, OR")
        print("   the symbol registration ordering in Numba differs.")
    else:
        print("Replay also fails. The reduced path uses the SAME llvmlite")
        print("resolver that Numba uses, with NRT_MemInfo_call_dtor freshly")
        print("registered via ll.add_symbol just before NewOrcJIT was built.")
        print("=> Strongly suggests llvmlite-side resolver wiring "
              "(NumbaSymbolGenerator / ProcessSymbolsJITDylib in "
              "ffi/neworcjit.cpp) is not surfacing add_symbol entries to "
              "ORC's symbol lookup.")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(3)
