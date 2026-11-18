/# OrcJIT migration -- implementation notes

- scope: `feature/orcjit` (numba) + unmerged llvmlite in `/llvmlite/ffi/neworcjit.cpp`, `/llvmlite/llvmlite/binding/neworcjit.py`
- two repos coupled: llvmlite engine implements semantics (content-hash rename, framed-blob cache, replay-skip); numba codegen reshaped around them
- sources: `./test.sh` (smoke catalogue), diffs vs `origin/main`

---

## 1. Why migrate

- MCJIT deprecated upstream
- old: `JITCPUCodegen` on `llvm::ExecutionEngine` -> flat symtab, eager finalize, `set_object_cache` hooks, in-process pointer lookup
- ORCv2/LLJIT is successor but different defaults
  - per-`JITDylib` scoping
  - lazy materialization
  - no flat-table contract
  - asserts on duplicate weak defs through IR layer

Two options for restoring MCJIT-shaped semantics:

- **JITDylib namespacing** (one dylib per library)
  - "correct" ORC design
  - behavioural change to `get_pointer_to_function`, `set_env`, `RuntimeLinker`
  - touches every cross-library caller in numba + every downstream (numba-cuda, numba-rocm, numba-dpex, user dispatchers)
- **Content-hash + flat namespace** (this branch)
  - translation layer, not a model change
  - `library.get_pointer_to_function(name) -> addr` unchanged
  - cross-process cache safety is structural: same content -> same renamed name
  - cost: cross-library runtime dedup of byte-identical helpers no longer at IR layer
    - in practice inliner has eaten hot helpers before they hit the JIT -> barely matters

- chose content-hash for non-invasive property
  - rename map lives in binding, not numba
  - every call site keeps its shape; binding silently translates via per-handle map

---

## 2. Architecture, end-to-end

```
                    +-----------------------------+
   Numba IR  --->   | CPUCodeLibrary.finalize()   |
                    |  - link_in helpers          |
                    |  - inliner + opt passes     |
                    |  - _finalize_final_module() |
                    |       _add_module(module)   |
                    +-------------|---------------+
                                  v
                    +-----------------------------+
                    | JitEngine.add_module(mod)   |  numba/core/codegen.py
                    +-------------|---------------+
                                  v
                    +-----------------------------+
                    | NewOrcJIT.add_ir_module     |  llvmlite/binding/neworcjit.py
                    +-------------|---------------+
                                  v
                    +-----------------------------+
                    | LLVMPY_NewOrcJIT_AddIRModule|  ffi/neworcjit.cpp (C binding)
                    |     -> parseAndCompileAndAdd
                    +-------------|---------------+
                                  v
                    +-----------------------------+
                    | OrcEngine::compileAndAdd    |
                    |  - setDataLayout            |
                    |  - renameModule (SHA-256)   |
                    |  - replay-skip probe (ES.lookupFlags)
                    |  - addPassesToEmitFile -> .o|
                    |  - frame blob (NORC|RM|obj) |
                    |  - JIT->addObjectFile       |
                    |  - notify cb (framed blob)  |
                    +-------------|---------------+
                                  v
                    +-----------------------------+
                    | ORC ObjectLinkingLayer      |
                    |   first-wins flags handle   |
                    |   weak/linkonce_odr dups    |
                    +-----------------------------+
```

Lookup path (mirror):

- caller always passes **original (pre-rename) symbol name**
- `JitEngine.get_function_address(name, handle=...)` -> `NewOrcJIT.get_function_address(name, handle=...)`
- binding consults `_get_rename_map(handle)` to translate
- C entry queries JITDylib by translated linker name

---

## 3. Use case catalogue (what `test.sh` verifies)

- `./test.sh` is the smoke suite gating the migration
- each UC pins a specific milestone or regression class
- listed as the IR/engine lifetime the script exercises

### UC1 -- basic `@njit` (`tryorcjit.py`)

- `@njit def foo(x): return x + 1; foo(123)`
- lifetime
  1. `JITCPUCodegen.__init__` builds `NewOrcJIT` (LLJIT); one process-symbol generator + one numba-resolver generator on main JITDylib
  2. numba lowers `foo` to `llvmlite.ir.Module`; helpers (NRT incref/decref, error infra) arrive via linking libs
  3. `CPUCodeLibrary.finalize()`
     - `link_in(helper._get_module_for_linking(), preserve=True)` per helper -> helper bodies as `linkonce_odr`
     - `_optimize_final_module()` cheap + full module-pass managers (inliner eats most closures)
     - `_finalize_final_module()` registers in `_libs_by_module_id`, then `_add_module`
  4. `OrcEngine::compileAndAdd`
     - set Module DataLayout from engine TargetMachine (codegen asserts on mismatch)
     - `renameModule`: SHA-256 every renamable GV, splice 16-hex tag (Itanium ABI tag for `_Z...`, `.<hex>` suffix otherwise); update intra-module refs via `GV::setName`
     - emit `.o` via `legacy::PassManager` + `addPassesToEmitFile`
     - `frameBlob`: NORC magic + version + u32 rename-map length + alternating null-terminated rename pairs + raw object bytes
     - `JIT->addObjectFile(RT, MemoryBuffer)` into main JITDylib
     - fire notify callback with framed blob (no-op when `cache=False`)
  5. `JITCodeLibrary._finalize_specific`
     - `_scan_and_fix_unresolved_refs(final_module, handle)`; no placeholders here; still records `foo -> handle` in `RuntimeLinker._defined`
     - `finalize_object()` -> `OrcEngine::finalizeObject` walks every handle's `DefinedNames` (deduped across handles), forces materialization via `ES.lookup`; codegen errors surface here (matches MCJIT contract)
  6. dispatcher calls `library.get_pointer_to_function("foo")`
     - `is_symbol_defined("foo", handle=h)` -> binding translates to `foo.<hash>` -> `LLVMPY_NewOrcJIT_IsSymbolDefined` (non-materializing `ES.lookupFlags`)
     - `get_function_address("foo", handle=h)` -> same translation -> `LLJIT::lookup` -> executor address

- variant: `tryorcjit_reduced.py` is two-phase: monkeypatches `JitEngine.add_module` to capture every IR module, replays through fresh `NewOrcJIT` with NRT helpers re-registered; used to disambiguate numba-side vs llvmlite-side regressions

### UC2 -- array / NRT path (`tryarray.py`, `tryarray2.py`)

- `np.arange(10)`-shaped JIT exercises NRT helper resolver chain
- NRT helpers (`NRT_incref`, `NRT_decref`, `NRT_MemInfo_*`, ...) registered into process symbol table via `ll.add_symbol(name, addr)` at NRT import
- engine main JITDylib has 2-deep generator chain
  - `NumbaSymbolGenerator` wraps numba Python-side resolver callback
  - `LLVMAddSymbolResolver` wraps `llvm::sys::DynamicLibrary::SearchForAddressOfSymbol` (sees both `ll.add_symbol` and process dlsym)
- misses fall through both generators in order; NRT helpers resolve via second
- `tryarray2.py` adds `bar()` calls in loop -> helper inlining + cross-function refs in one module

### UC3 -- on-disk object cache miss + hit (`trycache.py`)

- `@njit(cache=True) foo` calls `@njit(cache=True) bar`; run twice
  - first = miss (compile + write)
  - second = hit (replay)
- verifies single-library replay AND cross-library cached call
  - `foo`'s cached `.o` references `bar.<hash>`; `bar`'s cached `.o` replayed first so name resolves

- **miss path**
  1. compile as UC1
  2. `CPUCodegen._init` installed notify callback; engine fires it inside `compileAndAdd` (synchronous) with framed blob
  3. trampoline keys back into `_libs_by_module_id`, finds library by module identifier (= library name), calls `library._on_object_compiled(buf)`
  4. library stashes `buf` on `self._compiled_object`
  5. numba `FunctionCache` later calls `library.serialize_using_object_code()` -> `(name, 'object', (compiled_object, shared_bitcode))`; framed blob lands on disk

- **hit path**
  1. numba cache detects hit, calls `library._unserialize(codegen, ('object', (object_code, shared_bitcode)))`
  2. library
     - `_set_compiled_object(object_code)` so re-serialize emits same bytes
     - `_shared_module = ll.parse_bitcode(shared_bitcode)` -- used only for `_get_module_for_linking` if another library links this one; engine never sees bitcode on hit path
     - `engine.add_object_file(object_code, name=...)` straight into binding, **bypassing** rename pass (re-running renamer would re-hash and mismatch symtab)
     - `_scan_and_fix_unresolved_refs(shared_module, handle)` patches `.numba.unresolved$<sym>` placeholders
  3. `add_object_file` in C++
     - `parseFrame(blob)` extracts bundled rename map + inner object bytes; legacy (unframed) blobs tolerated as raw object bytes with empty rename map
     - walks `.o` symtab for `SF_Global && !SF_Undefined` -> `DefinedNames[H]`; strong-linkage subset -> `StrongDefs`
     - **replay-skip probe**: bulk `ES.lookupFlags` on MainJD for `StrongDefs`; full overlap -> install rename map, alloc empty `DefinedNames[H]`, return without `JIT->addObjectFile`; partial -> normal load
  4. `finalize_object()` materializes any new symbols

- cross-library cached call works because both `.o`s carry post-rename names
  - second `add_object_file` for `foo` finds `bar.<hash>` already in JITDylib (from `bar`'s cached `.o` moments earlier)
  - relocation resolves cleanly

### UC4 -- mutual recursion across cache (`trymutualrecur.py`)

- `foo` calls `bar` calls `foo`; run twice
- verifies `.numba.unresolved$<sym>` placeholder mechanism on cache-replay path

- live compile uses `RuntimeLinker.scan_unresolved_symbols`
  - LLVM external GV `.numba.unresolved$foo` is `add_global_mapping`'d to point at `nrt_unresolved_abort` slot
  - after both libraries finalize, `resolve()` walks unresolved set against defined-name map, patches each `c_void_p` slot to post-rename address
  - runtime indirect call goes through slot
- on cache hit
  - placeholder GV baked into `.o` as undefined external
  - replay-time `_scan_and_fix_unresolved_refs` against *shared bitcode* (which still carries decl) re-`add_global_mapping`s slot
  - `scan_defined_symbols` records this library's defined function names -> handle, so sibling library finalizing later resolves cross-library callees by name via handle rename map

### UC5 -- same-IR replay-skip in `compileAndAdd` (`q4_repro_min.py`, `q4_repro_numba.py`)

- Q4 regression: two `add_ir_module` calls with byte-identical IR produce same `foo.<hash>` post-rename
  - without replay-skip probe in `compileAndAdd`, second `JIT->addObjectFile` SIGSEGVs inside JITLink on duplicate strong external
- hit in practice by
  - `forceobj=True` whose body forces objmode looplift -> looplift drives second `_finalize_final_module` on byte-identical IR
    - `test_usecases.test_sum1d_pyobj`
    - `test_usecases.test_string_conversion`
    - `test_caching.test_looplifted`
  - any caller submitting same IR twice
- fix mirrors UC3's replay-skip
  - probe MainJD for strong defs
  - full overlap -> `ReplaySkip = true`
  - still emit `.o` and fire notify (`CodeLibrary._get_compiled_object` needs blob for on-disk serialization)
  - skip `JIT->addObjectFile`

### UC6 -- Q2 (hash-stable IR) + Q3a (same-engine reload)

- `trycacheissues.py`: two fresh compiles of array-overload-heavy module must produce byte-identical IR
  - Q2 (resolved) replaced `id(obj)` / `hash(bytes)` in pickle GV names + `insert_const_bytes` with SHA1 of payload
  - `hash(bytes)` was fragile -- PYTHONHASHSEED-randomized -> different GV names -> different post-rename hashes -> cache-miss on otherwise-identical modules
  - now byte-stable
- `tryforkissue_control.py`: Q3a regression
  - `SharedCacheLocator` directs two `@njit` wrappers to same cache dir
  - reader's cache-hit replay re-loads writer's `.o` into same engine where writer already published symbols
  - ORC would otherwise raise `DuplicateDefinition`; UC3 replay-skip path in `addObjectFile` suppresses redundant load

### UC7 -- Q5 fork-stable function UID (`tryforkissue.py`)

- bug fixed (narrowly): LLVM type-mismatch crash when parent compiles wrapper whose mangled name was already claimed by a *different* wrapper in forked child
- root cause
  - `bytecode.FunctionIdentity.from_function` constructed mangled symbol name with uid from `_unique_ids = itertools.count()` (process-global counter, once per `FunctionIdentity`)
  - `os.fork()` duplicates counter -> parent/child can produce same uid for different wrappers
  - same mangled name + different LLVM function types -> ORC `Function type returns {...} but resty={...}` crash
- fix: switch uid source to `id(func)`
  - mangled name now embeds wrapper address
  - parent's W_a and child's W_c live at unrelated heap addresses (independent heaps; even after fork, freshly-allocated objects differ) -> distinct mangled names -> no crash
  - `id(func)` is fork-stable in the relevant sense: pre-fork wrapper has same address in child immediately after fork (COW address space)
- layer model (each catches a different collision class)
  - L1: numba uid in mangled name (Q5: `id(func)`)
    - catches counter-desync after fork
    - misses coincidental same-`id()` across unrelated processes
  - L2: OrcJIT content-hash renamer
    - SHA-256 of IR body spliced into linker name post-mangle
    - catches same-name-different-body inside one JITDylib (cross-process replay alongside fresh compile)
    - closes the crash class no matter what L1 produces
    - misses cache-key collisions (lookup happens *before* renamer runs)
  - L3: numba cache key `(qualname, argtypes, signature, ...)`
    - catches functions distinguished by source (different `def` names in same module) -- 99% case
    - misses lambdas, exec-generated wrappers, factory closures, captured globals/closures whose value differs writer-vs-reader
- caveat at `numba/core/bytecode.py:from_function`
  - intra-process: `id()` is reused after GC; if dispatcher released wrapper and freed address reallocated to different wrapper compiled later in same compilation context -> L1 collision
  - L2 still saves engine from crash (different IR bodies, different content hashes)
  - L3 silent-wrong-answer mode latent
  - numba dispatcher pins wrapper for its lifetime -> intra-process variant does not fire
  - durable hardening recommended for less-controlled callers (per-process nonce XOR, `WeakValueDictionary` intern)

---

## 4. Symbol-naming taxonomy

- three orthogonal opt-outs/overrides apply to renaming
- policy encoded in `/llvmlite/ffi/neworcjit.cpp:shouldRename` + `hashGlobalValue`

| category | linkage | rename? | hash source | notes |
|---|---|---|---|---|
| function/GV def | external / available_externally | yes | own IR body | standard |
| function/GV def | linkonce_odr / weak_odr / common | yes | own IR body | first-wins at object layer |
| declaration (extern) | any | no | -- | cross-module refs; resolved post-rename through other modules' maps |
| local (internal/private) | local | no | -- | never enters JITDylib symtab; pickle GVs (`.const.*`) |
| `!numba.preserve`-tagged | any non-local | no | -- | looked up by linker name from outside the JIT: NRT helpers, externs via `ll.add_symbol` |
| `!numba.env_for`-tagged GV | typically `common` | yes | **referenced function's IR body** | numba env GVs (`_ZN08NumbaEnv...`); body uniform (`null voidptr`) so self-hash would collapse every env to one name; delegation binds env identity to function identity; resolves Q1 |

- numba-side audit obligations (Step 5)
  - forward: every symbol numba looks up by linker name from outside owning module must be `numba.preserve`-tagged; otherwise rename rewrites it and lookup misses
  - inverse: every `numba.preserve`-tagged symbol's name must uniquely identify its underlying thing across all libraries coexisting in one JITDylib
    - env GVs were the cautionary tale (same `env_name` for different Environment objects across libraries)
    - resolved by switching to `!numba.env_for`
- current preserve clients after audit
  - NRT helpers (`nrt_unresolved_abort`, `NRT_*`)
  - other externally-named symbols referenced from C
  - env GVs + pickle GVs intentionally removed

---

## 5. Framed-blob cache format

```
+---------+---------+---------------+----------------------+--------------+
| 'NORC'  | ver=1   | u32 LE RM_LEN | rename map bytes     | object bytes |
| 4 bytes | 1 byte  | 4 bytes       | RM_LEN bytes         | rest         |
+---------+---------+---------------+----------------------+--------------+
```

- rename map bytes: alternating null-terminated `<old>\0<new>\0...` (same as `LLVMPY_NewOrcJIT_GetRenameMap` output)
- legacy unframed bytes (no `NORC` magic) tolerated -- `parseFrame` returns `HasFrame=false` and treats whole blob as object bytes with empty rename map
- bad version: hard error
- one artifact, not two parallel ones
  - `serialize_using_object_code` returns framed blob verbatim
  - `_unserialize` hands it back to `add_object_file` which installs rename map automatically

---

## 6. Cross-repo interdependencies

- cherry-picking only one half breaks specifically:

| llvmlite feature | numba consumer | what breaks if missing |
|---|---|---|
| `create_new_orcjit()` | `CPUCodegen._init` | engine swap; can't replace `create_mcjit_compiler` |
| `target_data` property | `_init`, `_data_layout` | Module DataLayout unset -> opt/codegen asserts |
| `add_ir_module` returns handle | `JITCodeLibrary._jit_handle` | all lookups become flat-name -> wrong symbols on collision |
| per-handle rename map (internal) | `_get_rename_map`, `_translate_name` | `get_pointer_to_function` returns 0 on every renamed symbol |
| `is_symbol_defined(name, handle=)` | `JITCodeLibrary.get_pointer_to_function` | misses or hangs on materialization |
| `get_function_address(name, handle=)` | `RuntimeLinker.resolve`, `set_env` | cross-library calls / env patching can't find target |
| framed-blob notify | `_on_object_compiled` | cache miss captures nothing -> "no compiled object yet" |
| `add_object_file` parses frame + installs map | `_unserialize` (kind='object') | replay loses rename map -> all lookups via handle miss |
| `addObjectFile` replay-skip (UC3, UC6) | `_unserialize` on shared cache locator | ORC `DuplicateDefinition` on second load |
| `compileAndAdd` replay-skip (UC5) | `_finalize_final_module` on forceobj+looplift | SIGSEGV in JITLink on duplicate strong defs |
| `finalizeObject` dedups across handles | `JITCodeLibrary._finalize_specific` | ORC asserts "Duplicate dependence notification?" when same `helper.<hash>` is in multiple handles' DefinedNames |
| `!numba.env_for` recognition | `BaseContext.declare_env_global(referencer_name=)` | env GVs collapse to one name -> JITDylib collision |
| `!numba.preserve` recognition | NRT, externs | renamer rewrites NRT helper names -> C side can't find them |
| `LLVMAddSymbolResolver` (dlsym) | NRT `ll.add_symbol` calls | NRT helpers unresolved -> `Symbols not found: NRT_*` |
| `add_global_mapping` accepts string-or-GV | `JitEngine.add_global_mapping` for `.numba.unresolved$*` | resolved-pointer slots don't get registered |

- numba-side expectations consumers must preserve
  - `library.get_pointer_to_function(name)` keyed on IR name, not linker name
    - callers holding flat linker-name view (e.g. `get_global_value_address` results re-looked-up by string) need handle context to translate
  - `library.set_env(env_name, env)` replaces old `Codegen.set_env` shim
    - env name is the original; library-scoped because each library has own rename map
  - `library._jit_handle` is the handle for every name lookup against this library
    - threaded through `JITCodeLibrary`, `RuntimeLinker._defined`, `JITCodeLibrary._unserialize`

---

## 7. What was removed / what behaviour changed

### Removed (vs MCJIT-targeting numba)

- `JitEngine._defined_symbols` Python-side mirror set + `_load_defined_symbols` helper
  - replaced by binding's `is_symbol_defined` (non-materializing JITDylib query)
- `Codegen.set_env` -> renamed `JITCodeLibrary.set_env`, library-scoped
  - caller in `compiler.py:CompileResult._rebuild` and `cpu.py:_get_executable_form` adjusted
- IR-side `ObjectCache` plumbing in binding
  - replaced by framed-blob notify trampoline
  - `set_object_cache(get=...)` accepted but ignored
- C++-side `demoteAlreadyDefinedDuplicates` pass that mimicked RuntimeDyld first-wins by rewriting duplicate weak defs to external decls
  - object-layer first-wins (`setOverrideObject...` + `setAutoClaim...`) does this natively

### Behaviour changed

- eager finalize is opt-in via `finalize_object()`
  - ORC lazy by default; engine emulates MCJIT eager semantics by `ES.lookup` over every handle's defined names
  - errors surface here
- on-disk cache is content-addressed structurally
  - two runs producing byte-identical IR (post Q2) produce byte-identical framed blobs
  - cross-process cache hits "just work" provided module identifier is stable (numba sets to library name)
- `get_pointer_to_function` requires handle for renamed symbols
  - MCJIT-era flat-name lookup still works for preserve-tagged symbols (NRT helpers)
  - anything numba defines in user IR must be looked up handle-aware
- static/local-linkage symbols never renamed
  - never enter JITDylib external table
  - pickle GVs being internal-linkage means renamer skips them
  - SHA1-of-payload name (Q2) needed for byte-identical-IR purposes, not collision avoidance
- `Module::data_layout` set by engine inside `compileAndAdd`, not by numba `_optimize_functions`
  - old unconditional assignment removed (commented out) -- comment kept so reviewers don't reintroduce

---

## 8. Quick navigation

| concern | file |
|---|---|
| engine lifecycle, renamer, replay-skip | `/llvmlite/ffi/neworcjit.cpp` |
| Python binding, handle-aware lookups | `/llvmlite/llvmlite/binding/neworcjit.py` |
| engine swap, `JitEngine`, `_libs_by_module_id` | `numba/core/codegen.py` (`CPUCodegen._init`, `JitEngine`) |
| library-scoped lookup + env patcher | `numba/core/codegen.py` (`JITCodeLibrary`) |
| cache replay (object kind) | `numba/core/codegen.py` (`CPUCodeLibrary._unserialize`) |
| recursive-call placeholders | `numba/core/codegen.py` (`RuntimeLinker`) |
| env GV metadata | `numba/core/base.py` (`declare_env_global`) |
| env-tag callers | `numba/core/{lowering,callwrapper,cpu}.py`, `numba/np/ufunc/wrappers.py` |
| pickle GV / bytes content-hash names | `numba/core/pythonapi.py`, `numba/core/base.py` (`insert_const_bytes`) |
| fork-stable uid | `numba/core/bytecode.py` (`FunctionIdentity.from_function`) |
| smoke-test suite | `numba/test.sh` and the `try*.py` / `q4_*.py` scripts |
| engine unit tests | `/llvmlite/testorcjit.py` |
| gaps + test-motivation notes | `numba/orcjit_gaps.md` |
