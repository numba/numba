set -xe

# Smoke tests for the OrcJIT migration (feature/orcjit).
# See CLAUDE.md for context; each group below pins a specific milestone
# or regression class from the migration plan.

# Test 1 — Basic OrcJIT compile + execute.
# tryorcjit.py: minimal @njit smoke (foo(123) == 124).
# tryorcjit_reduced.py: two-phase IR capture/replay through a fresh
# NewOrcJIT, disambiguates Numba-side vs llvmlite-side regressions.
python tryorcjit.py
python tryorcjit_reduced.py

# Test 2 — Array / NRT path through OrcJIT.
python tryarray.py

# Test 3 — Second array shape; exercises additional NRT helpers.
python tryarray2.py

# Test 4 — On-disk object cache (framed-blob, NORC header).
# Run twice: first is miss (compile + write), second is hit (replay).
# Includes cross-library cached call (foo → bar resolved from cache).
rm -rf __pycache__
NUMBA_DEBUG_CACHE=1 python trycache.py
NUMBA_DEBUG_CACHE=1 python trycache.py

# Test 5 — Mutual recursion across the cache boundary.
# Exercises RuntimeLinker handle-aware resolve + cache-replay
# _scan_and_fix_unresolved_refs (.numba.unresolved$<sym> placeholders).
rm -rf __pycache__
NUMBA_DEBUG_CACHE=1 python trymutualrecur.py
NUMBA_DEBUG_CACHE=1 python trymutualrecur.py

# Test 6 — Q4 regression: duplicate strong defs in compileAndAdd.
# q4_repro_min.py: pure llvmlite IR-side replay-skip.
# q4_repro_numba.py: forceobj + looplift drives two finalizes of the
# same IR; second addObjectFile must not SIGSEGV.
python q4_repro_min.py
python q4_repro_numba.py

# Test 7 — Q2 (hash-stable IR) + Q3a (same-engine cache reload).
# trycacheissues.py: two compiles produce byte-identical IR; cache
# miss + hit both succeed.
# tryforkissue_control.py: same-process SharedCacheLocator replay
# through addObjectFile replay-skip (no fork).
NUMBA_DEBUG_CACHE=1 python trycacheissues.py
NUMBA_DEBUG_CACHE=1 python tryforkissue_control.py

# Test 8 — Q5: fork-stable FunctionIdentity uid (id(func) instead of
# itertools counter). Parent + child agree on mangled_name across fork.
NUMBA_DEBUG_CACHE=1 python tryforkissue.py

echo "ALL PASSED"
