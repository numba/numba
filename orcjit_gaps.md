# OrcJIT migration -- gaps + test motivation

- companion to `orcjit_report.md` (implementation/design)
- covers: why each test exists (what regression class it pins), and the gaps still open after the migration
- terse dev-notes form

---

## 1. Why the smoke tests exist

- `./test.sh` is not coverage; it's a regression catalogue
- each script encodes one specific failure mode hit during the migration
- listed by the bug class they pin, not the API surface they touch

### UC1 -- `tryorcjit.py` / `tryorcjit_reduced.py`

- baseline: does an `@njit` even compile+execute on the new engine
- `_reduced` variant exists to bisect numba-side vs llvmlite-side regressions
  - captures every IR module via monkeypatched `JitEngine.add_module`
  - replays through fresh `NewOrcJIT` with NRT helpers re-registered
  - if `_reduced` passes but `tryorcjit.py` fails -> bug is numba-side
  - if both fail -> llvmlite-side

### UC2 -- `tryarray.py`, `tryarray2.py`

- pins NRT helper resolution
- if either generator chain breaks (`NumbaSymbolGenerator`, `LLVMAddSymbolResolver`) NRT symbols go unresolved
- `tryarray2.py` adds helper inlining + cross-function refs in one module (catches generator that resolves single-call but not loop-body)

### UC3 -- `trycache.py`

- pins on-disk cache miss+hit + cross-library cached call
- miss path verifies notify-callback fires synchronously inside `compileAndAdd` (Numba captures blob via `_on_object_compiled`)
- hit path verifies `add_object_file` bypasses renamer (re-running renamer would re-hash and mismatch symtab)
- cross-library cached call verifies that `bar`'s post-rename name in `foo`'s `.o` resolves against `bar`'s separately-cached `.o` loaded moments earlier

### UC4 -- `trymutualrecur.py`

- pins `.numba.unresolved$<sym>` placeholder mechanism on cache-replay path
- live compile uses `RuntimeLinker.scan_unresolved_symbols`; trivial
- replay is the interesting case: placeholder GV is baked into `.o` as undefined external; without `_scan_and_fix_unresolved_refs` against shared bitcode on replay, the slot is never `add_global_mapping`'d -> indirect call aborts

### UC5 -- `q4_repro_min.py`, `q4_repro_numba.py`

- pins same-IR replay-skip inside `compileAndAdd`
- two `add_ir_module` calls with byte-identical IR -> same `foo.<hash>` post-rename -> JITLink SIGSEGV on duplicate strong external without replay-skip
- hit in practice by `forceobj=True` + looplift; affected
  - `test_usecases.test_sum1d_pyobj`
  - `test_usecases.test_string_conversion`
  - `test_caching.test_looplifted`
- `q4_repro_min.py` is pure-llvmlite (~12 lines IR), `q4_repro_numba.py` is a real forceobj loop

### UC6 -- `trycacheissues.py`, `tryforkissue_control.py`

- two regression classes wedged together
- `trycacheissues.py` pins **byte-identical IR across two fresh compiles** (Q2)
  - `id(obj)` / `hash(bytes)` in pickle GV names + `insert_const_bytes` was PYTHONHASHSEED-randomized
  - different GV names -> different post-rename hashes -> cache miss on otherwise-identical modules
  - Q2 replaced with SHA1 of payload
- `tryforkissue_control.py` pins **same-engine cache reload** (Q3a)
  - `SharedCacheLocator` routes two wrappers to same cache dir
  - reader's hit-replay re-loads writer's `.o` into engine where writer already published symbols
  - ORC would raise `DuplicateDefinition`; replay-skip in `addObjectFile` suppresses the redundant load

### UC7 -- `tryforkissue.py`

- pins fork-stable UID (Q5)
- root cause: `_unique_ids = itertools.count()` duplicated across `os.fork()`
- writer (child) and reader (parent) could produce same uid for different wrappers -> same mangled name + different LLVM function types -> ORC type-mismatch crash
- fix: derive uid from `id(func)` instead of counter
- `id()` is fork-stable in the relevant sense (pre-fork wrapper has same address in child immediately after fork)
- test exercises both control branch (no fork, same engine) and fork branch

---

## 2. Open items

- Step 5 audit (forward + inverse `numba.preserve`)
  - forward seems complete after Q1
  - inverse not yet documented
- debug-print hygiene
  - both numba (`NUMBA_ORCJIT_DEBUG`) and llvmlite (`LLVMLITE_ORCJIT_DEBUG`) gated, off by default
  - decision left for production: keep gated, or delete outright
  - current recommendation: keep -- load-bearing for ongoing work
- `config.ENABLE_PROFILING` / `enable_jit_events`
  - ORC has GDB/perf event hooks matching MCJIT
  - not yet wired; low priority
- `id(func)` GC-reuse hazard for `FunctionIdentity.unique_id`
  - durable hardening (per-process nonce XOR, `WeakValueDictionary` intern) not yet applied
  - not exercised by current tests
- cross-process cache key policy
  - cache key is module identifier; numba sets to library name
  - for cross-process cache hits with different library-name spelling but identical content, callers would need content-derived identifier
  - numba-side policy decision, not blocking

---

## 3. Cache-key gap: `tryforkissue.py`-class collisions (silent-wrong-answer)

- UC7 closes the **crash** class
- silent-wrong-answer class remains: two genuinely different functions aliasing on `(qualname, argtypes)` and sharing a cache directory
- upstream of OrcJIT migration; not implemented on this branch
- options below are sketches for follow-up cache-layer work

### 3.1 Three-layer collision model

| layer | mechanism | catches | misses |
|---|---|---|---|
| L1 numba `uid` in mangled name (Q5: `id(func)`) | per-function tag in linker-name *before* engine sees IR | counter-desync after fork (deterministic fork-collision class) | coincidental same-`id()` across unrelated processes (spawn, fresh interpreter); rarer but real, no cross-process guarantee |
| L2 OrcJIT content-hash renamer | SHA-256 of IR body, spliced post-mangle | same-name-different-body inside one JITDylib (cross-process replay alongside fresh compile of different body) | cache-key collisions; lookup happens before renamer runs, so renamer cannot reach into cache layer |
| L3 numba cache key `(qualname, argtypes, signature, ...)` | decides hit/miss before any compilation or rename | functions Python distinguishes at source level (different `def` names in same module) -- 99% case | lambdas (`<lambda>`), exec-generated wrappers (all carry inner function name), factory closures producing differently-behaving functions under one `__name__`, captured globals/closure cells whose value differs writer vs reader |

- Q5 fix lives entirely at L1
- L2 is load-bearing for crash class regardless of L1
- L3 gap is above both, unaddressed by this branch

### 3.2 The crash Q5 fixed (narrowly)

```
Proc1 (parent, pid=100):
  1. two @njit wrappers W_a, W_b back-to-back
  2. _unique_ids returns 1, then 2
       FunctionIdentity(W_a).uid = 1  -> mangled "...W_a$1..."
       FunctionIdentity(W_b).uid = 2  -> mangled "...W_b$2..."
  3. fork() -> Proc2 inherits counter state

Proc2 (child, pid=200):
  4. defines different wrapper W_c (e.g. PyTensor exec-factory; shares
     qualname with W_a but different inner impl / return type)
  5. _unique_ids.next() in child also produces uid=1
       FunctionIdentity(W_c).uid = 1  -> mangled "...W_c_qualname$1..."
  6. child compiles W_c, engine publishes "cfunc...W_c_qualname$1..."
     in JITDylib; writes cache entry for W_c under (qualname, argtypes)
  7. child exits

Proc1 (parent), later:
  8. type-inference triggers compile of fresh W_a' (or cache replay)
  9. mangled name also "...<qualname>$1..." (qualname matches and
     parent's counter independently produced 1 at some point)
 10. cached .o being loaded carries strong symbol with same mangled
     name but different LLVM function type (W_c's 3D->4D vs W_a's
     3D->3D). ORC: "Function type returns {...} but resty={...}". crash
```

Two ingredients

- linker symbol name (post-mangle, pre-content-hash) shared across two different functions because counter is order-dependent and desyncs after fork
- ORC checks function types on duplicate strong symbol defs, raises rather than silently first-wins

How `id(func)` dissolves it

```
With uid = id(func):
  - Proc1's W_a at addr A_a -> mangled "...W_a$A_a..."
  - Proc2's W_c at addr A_c -> mangled "...W_c_qualname$A_c..."
  - A_a != A_c (independent heaps; even post-fork the *parent's
    pre-existing W_a* and the *child's freshly allocated W_c* differ)
  - two distinct mangled names -> two distinct strong symbols -> no crash
```

- mechanical win: uid as function of `id(func)` is not order-dependent; parent + child don't drift into shared mangled names just from similar compile counts
- `id(func)` fork-stable in the narrow sense: wrapper that existed in parent before fork has same address in child immediately after fork (COW)
  - same pre-fork wrapper -> agree on uid
  - different wrappers -> disagree (correct)

### 3.3 Spawn / unrelated-process collision: probabilistic, same crash shape

```
Proc1 (writer, spawn):
  - W_a at 0x7fA1B0; mangled "...fn$0x7fA1B0..."
  - IR body hashes to H_a; post-rename "...fn$0x7fA1B0...B17_H_a..."
  - cache key (qualname="fn", argtypes=(int32,)) -> .o with that symbol

Proc2 (reader, spawn):
  - W_c (semantically different: returns float64) by coincidence
    allocated at SAME address 0x7fA1B0
  - mangled "...fn$0x7fA1B0..." -- byte-identical to Proc1's at L1
  - cache lookup hits Proc1's entry; loads its .o
  - Proc2 also freshly compiles W_c (another overload path)
    W_c body hashes to H_c != H_a; post-rename "...fn$0x7fA1B0...B17_H_c..."
  - two strong symbols in Proc2's JITDylib:
       from cached .o: ...B17_H_a (i32) -> i32
       from fresh IR:  ...B17_H_c (i32) -> f64
    Different names thanks to L2; NO type-mismatch crash.
  - L2 saved the engine; L3 did not save the program.
```

- L2 closes crash class regardless of what L1 produces
- even when uids collide by coincidence, body-hash disambiguates linker symbols inside engine
- this is why Q5's narrow fix is enough to take the spawn shape from "deterministic crash" to "no crash"

### 3.4 The silent-wrong-answer mode (L3)

- L2 does NOT reach the cache lookup that already happened *before* renamer ran
- in spawn case, if Proc2 only loads Proc1's cached `.o` and never freshly compiles a colliding overload, no symbol-table collision occurs inside Proc2's engine
- but Proc2 just executed W_a's body when user asked for W_c
- wrong answer, no warning

`tryforkissue.py` constructs the exact shape that exposes L3:

- two wrappers (`returns_3d`, `returns_4d`) sharing `qualname` `"fn"` (both produced by `exec()` of `def fn(...): ...`)
- routed to same cache directory by `SharedCacheLocator`
- different inner implementations and return types

Cache key `(qualname="fn", argtypes=...)` does NOT capture:

- module globals captured by function body
- free variables / closure cells
- identity of inner function the wrapper dispatches to

Contrast: `trycacheissues.py` shows L3 is fine for source-defined case

- `f3` and `f4` live in one file with identical decorators + signatures, but qualnames differ
- cache key separates them on qualname
- strip qualname distinction (exec wrappers, lambdas with `<lambda>` for all) -> L3 collapses to "every function with these argtypes maps to one cache entry"

- property of numba cache layer, not OrcJIT migration; MCJIT had same gap
- Q5 closes only crash class
- captured-state class remains a known limitation behind `@njit(cache=True)` regardless of JIT engine

### 3.5 Remaining intra-process hazard

- `id()` reused after GC
- if dispatcher released its wrapper and freed address reallocated to *different* wrapper compiled later in same compilation context -> L1 collision
- L2 still saves engine from crash (different IR bodies, hence different content hashes)
- L3 silent-wrong-answer still latent
- numba dispatcher pins wrapper for its lifetime -> intra-process variant does not fire
- durable hardening recommended (per-process nonce XOR, `WeakValueDictionary` intern)
- caveat at `numba/core/bytecode.py:from_function`

### 3.6 Why "just content-hash the cache key" doesn't work

- impulse: same IR -> same key, different IR -> different key
- fails for practical reason: cache key must be computable *before* compilation
- whole point of cache hit is to avoid compiling -> IR doesn't exist yet at lookup time
- bytecode-hash isn't a clean substitute
  - closures + globals are read at bytecode-execution layer, not visible in bytecode object itself
  - type inference depends on argtypes (cache already keys on them)
  - dispatch chains into other `@njit` functions whose IR is also lazy
- content-derived key would require simulating front half of compilation -> defeats purpose
- OrcJIT content-hash works because it runs *after* IR is in hand and *inside* engine
  - disambiguates symbols in JITDylib that already exists
  - cannot reach upstream into cache-key decision

- real options: "narrow admissible inputs" or "let caller carry the burden"

### 3.7 Option A -- ban dynamic-factory caching

- detect cases where cache key is provably insufficient; refuse to cache
- detectable signals
  - non-source-pinned `qualname`
    - heuristic `func.__qualname__` != `func.__name__` weak
    - harder: `inspect.getsourcefile(func)` returns real file AND `inspect.findsource(func)` succeeds with bytecode pointing at that file; `exec`-ed functions fail this
  - closure cells / module-global refs not captured in cache key
    - inspect `func.__closure__` for non-empty cells
    - walk `func.__code__.co_names` for `LOAD_GLOBAL` opcodes referencing names whose binding is mutable in writer's module
- policy: if either fires under `@njit(cache=True)`, raise at decoration time
- trade-offs
  - pro: closes class structurally; no silent-wrong-answer
  - pro: diagnostic is actionable
  - con: PyTensor and similar rely on dynamic-factory pattern; hard ban would break them
  - con: closure/global detection is conservative; false positives need opt-out or hand-curated whitelist that rots

### 3.8 Option B -- user-managed cache key

```python
@njit(cache=True, cache_key="dimshuffle:(2,0,1):f8")
def fn(x): ...
```

or programmatic

```python
fn = njit(cache=True, cache_key=lambda f: hash_of_pytensor_op_spec(f))(wrapper)
```

- cache key becomes `(qualname, argtypes, user_key, signature, ...)`
- already exists internally as `SharedCacheLocator.get_disambiguator()` (used for directory routing in `tryforkissue.py`)
- change is to elevate from "directory selector" to actual cache-key component, document it
- trade-offs
  - pro: doesn't break dynamic-factory users; gives them a knob
  - pro: composes with Option A (cache_key= lifts the ban)
  - pro: keeps numba out of guessing whether a capture is pure
  - con: foot-gun if callers pass weak keys; PyTensor would need audit
  - con: doesn't help users who don't know they're affected (`@njit(cache=True)` on exec-lambda with no `cache_key` still silently corrupts)

### 3.9 Option C -- layered: ban by default, opt out via user key

- under `@njit(cache=True)`
  1. run Option-A detectors
  2. if they fire and no `cache_key` supplied -> raise
  3. if `cache_key` supplied -> include in cache key and proceed
- `tryforkissue.py` becomes working program
  - `SharedCacheLocator` routes via per-op key; that key becomes `cache_key=`
  - detectors fire on exec'd wrapper, would have raised, but `cache_key` opt-out lifts ban
- default users get hard error instead of silent wrong answer
- power users get explicit documented escape hatch
- trade-offs
  - pro: strictly safer default
  - pro: existing safe users (`@njit(cache=True)` on normal `def`) see no change
  - con: PyTensor needs one-line `cache_key=` addition (survivable)
  - con: numba commits to maintaining detector heuristics

### 3.10 Implementation sketch (Option C)

- touch points
  - `numba/core/caching.py:CacheImpl._locator_classes`
    - `_CacheLocator.from_function` already returns disambiguator
    - promote disambiguator into cache key (currently used only for directory routing)
  - `numba/core/dispatcher.py:_Compiler` (or wherever `cache=True` honoured)
    - plumb `cache_key=` from decorator down to locator
  - `numba/core/serialize.py` / cache-key tuple
    - extend with `user_disambiguator` field
  - detector heuristics: new module `numba/core/cache_safety.py`
    - `is_dynamically_generated(func)`
    - `has_unstamped_captures(func)`
    - wire into `@njit(cache=True)` entry point with `NumbaCacheUnsafeError`
- back-compat
  - gate detectors behind `NUMBA_CACHE_SAFETY=warn|error|off`
  - default `warn` for one release, flip to `error` next release

### 3.11 Recommendation

- Option C
  - right default: cannot silently miscompile under `cache=True`
  - escape hatch small enough for real factory libs to adopt
  - implementation contained in cache layer, no engine-side changes
- neither Q5 nor OrcJIT content-hash can substitute
  - they live one layer below the cache decision
  - by the time they have something to say, wrong entry already chosen on disk
