# Comprehensive Rumba Migration Plan

## Summary

`rumba` is a new top-level package whose purpose is to reimplement the useful
core of Numba in Rust. It must not import, call, or depend on `numba`; Numba is
only a reference for bytecode handling, supported semantics, NumPy overload
behavior, diagnostics, and compatibility tests.

The long-term ownership model is Rust-first:

- Rust owns the public extension module through PyO3.
- Rust owns dispatcher state, signature specialization, bytecode decoding,
  Rumba AST construction, type inference, lowering, code generation, cache
  metadata, compiler invocation, diagnostics, and runtime ABI handling.
- There are no Python implementation files under `rumba/python/rumba`.
- Python remains only the consumer language and the language used by public API
  tests. Optional examples should live in documentation snippets rather than
  checked-in Python implementation files.

The target compiler pipeline is:

```text
Python function object
  -> Rust/PyO3 reads CPython code object and metadata
  -> Rust CPython bytecode decoder
  -> Rust-owned Rumba AST
  -> Rust-owned typed Rumba AST
  -> Rust generated C source
  -> Rust compiler driver builds shared library
  -> Rust/PyO3 dispatcher invokes compiled artifact
```

## Current Implementation

- Added `rumba/` as an independent `maturin` project with a PyO3 extension
  module named `rumba`.
- Added Rust-owned `rumba.__version__`, `rumba.njit`, and `rumba.jit`.
- Added a PyO3 dispatcher object with `.py_func`, `.signatures`, `_compiled`,
  `.inspect_bytecode()`, `.inspect_rumba_ast()`, and `.inspect_c()`.
- Added lazy compile-on-first-call for scalar signatures.
- Moved scalar type handling, C emission, cache key creation, generated-source
  writing, C compiler discovery, C compiler invocation, and shared-library
  artifact metadata into Rust.
- Moved decorator ergonomics, argument type discovery, exception types, compiled
  library loading, and native invocation into Rust.
- Removed `rumba/python/rumba`; no Python files implement the package.
- Removed the optional Python example script to keep the repository's Python
  footprint limited to tests.
- Added clear Rust-defined `RumbaUnsupportedError` and
  `RumbaCompilationError` exception types.

The first implemented subset supports scalar `int64`, `float64`, and `bool`
arguments, arithmetic, comparisons, simple `if` statements, `range` loops,
local assignment, augmented assignment, and scalar returns. NumPy array lowering
is intentionally not implemented yet.

## Non-Negotiable Direction

Rumba is not a Python reimplementation with a Rust helper. Rumba is a Rust
compiler/runtime exposed to Python.

The following pieces are now Rust-owned in the initial slice:

- Argument type discovery and signature construction.
- Dispatcher object and specialization cache.
- Public exception classes.
- Scalar type inference and unsupported-operation diagnostics for the current
  source/AST bridge.
- ABI conversion and native invocation for the currently supported homogeneous
  scalar signatures.
- Cache key generation and shared-library compilation.

The following pieces still need deeper Rust implementations:

- CPython bytecode reading from function code objects.
- Rust-owned Rumba AST detached from Python `ast` objects.
- Full typed AST and all inspection/debug representations.
- ABI argument conversion, including scalar and NumPy array views.
- Cache metadata serialization and invalidation.

## Public API

Initial public API:

- `rumba.__version__`
- `rumba.njit(fn=None, **options)`
- `rumba.jit(fn=None, **options)`

Supported options for now:

- `cache=False`
- `debug=False`
- `signature=None`

Unsupported Numba options must raise `RumbaUnsupportedError` instead of being
silently accepted or falling back to Python.

## Architecture Milestones

### Milestone 1: Importable Rust Extension

Status: Implemented.

Acceptance criteria:

```bash
cd rumba
maturin develop
python -m pytest
python -c "import rumba; print(rumba.__version__)"
```

### Milestone 2: Minimal User API

Status: Implemented in Rust.

The current API supports `@rumba.njit`, `@rumba.njit(...)`, and `rumba.jit` as
an alias. The dispatcher class is a PyO3 `#[pyclass]`.

### Milestone 3: Rust-Owned Scalar Codegen Path

Status: Implemented initial slice.

Rust now owns scalar C source generation, cache key creation, generated-source
writing, C compiler selection, C compiler invocation, and artifact metadata.

Remaining work:

- Split the current single Rust file into frontend, IR, typing, codegen,
  runtime, dispatcher, and diagnostics modules.
- Expose compiler command and cache path through inspection helpers.

### Milestone 4: Rust Bytecode Frontend

Status: Not started.

Replace the current Python source/AST bridge with a Rust frontend that reads
CPython code objects through PyO3 and decodes supported Python 3.10+ bytecode.

Required capabilities:

- Decode instructions, offsets, constants, names, locals, freevars, and source
  locations.
- Build control-flow blocks and stack effects.
- Reject closures, generators, exceptions, comprehensions, object operations,
  and unsupported opcodes with structured diagnostics.
- Produce a Rust-owned Rumba AST for straight-line code, branches, simple
  loops, `range`, `len`, scalar locals, and array indexing.
- Keep bytecode frontend tests independent from Numba imports.

### Milestone 5: Rust Dispatcher And Runtime Invocation

Status: Partial.

Implement the dispatcher as a PyO3 class.

Required capabilities:

- Store the original Python function as `.py_func`. Implemented.
- Discover argument types in Rust. Implemented for scalars.
- Compile lazily on first call for an observed signature. Implemented.
- Cache compiled artifacts by bytecode hash, constants, closure-free globals
  used, signature, Python version, Rumba version, target platform, and compiler
  flags. Partial.
- Invoke compiled functions from Rust instead of Python `ctypes`. Implemented
  for homogeneous scalar signatures up to three arguments.
- Preserve inspection helpers for bytecode, Rumba AST, typed AST, generated C,
  compiler command, and cache path. Partial.

### Milestone 6: Rust Type Inference And Diagnostics

Status: Partial scalar support inside Rust C emitter.

Promote the current ad hoc scalar typing into a dedicated Rust type inference
pass over the Rumba AST.

Required capabilities:

- Scalar types: `int64`, `float64`, `bool`.
- 1D array view types for contiguous `int64` and `float64` NumPy arrays.
- Clear ambiguity errors and unsupported-operation diagnostics.
- Diagnostic spans tied to bytecode offsets and source line information when
  available.

### Milestone 7: NumPy Array Interop In Rust

Status: Not started.

Planned first array support:

- 1D contiguous `int64` and `float64` arrays.
- Rust/PyO3 validation of dtype, dimensionality, contiguity, and mutability.
- ABI representation containing data pointer, length, item size, and stride.
- Lower array element load/store and `len(array)` to C.
- Support scalar-returning kernels that mutate arrays in place.

### Milestone 8: Compatibility Growth

Status: Not started.

Add supported features deliberately, with Rust implementation and tests for
each:

- Selected `math` scalar functions.
- Selected NumPy scalar functions.
- Simple reductions.
- `np.empty` for supported dtypes and one-dimensional shapes.
- Limited broadcasting only after array views and scalar loops are stable.

### Milestone 9: Packaging And Developer Workflow

Status: Partial.

Keep the project independently buildable:

```bash
cd rumba
maturin develop
python -m pytest
```

The source-tree fallback used during development may symlink a Cargo-built
extension for local testing, but `maturin develop` is the supported workflow.

## Explicitly Unsupported

Unsupported behavior must raise `RumbaUnsupportedError`, not fall back to
Python:

- Python objects beyond supported scalars and planned array views.
- Lists, dicts, sets, tuples, exceptions, comprehensions, generators, closures,
  recursion, object mode, GPU, parallel mode, and general NumPy broadcasting.
- Array-returning functions in the initial native path.
- Windows native compilation until the shared-library and compiler-driver path
  is stable.

## Test Plan

- Package tests for import, version, decorator forms, and unsupported options.
- Rust unit tests for bytecode decoding, Rumba AST construction, type inference,
  C generation, cache keys, and compiler command construction.
- Frontend tests for simple arithmetic, branches, loops, `range`, `len`, and
  array indexing.
- Execution tests for scalar arithmetic, branches, loops, cache reuse, and
  distinct signatures.
- NumPy tests for 1D array reads/writes, dtype mismatch, non-contiguous arrays,
  unsupported dimensions, and unsupported returns.
- Compatibility tests compare against Python first and against Numba only for
  explicitly supported semantics. Rumba implementation code must not import
  Numba.

## Progress Log

| Date | Update |
| ---- | ------ |
| 2026-04-26 | Initial migration plan created. |
| 2026-04-26 | Added independent `rumba/` package scaffold and first scalar C/ctypes execution slice. |
| 2026-04-27 | Moved scalar C generation, cache key generation, generated source writing, compiler discovery, compiler invocation, and artifact metadata into Rust. |
| 2026-04-27 | Updated project direction: Rumba is a Rust compiler/runtime with Python only as a thin package boundary and temporary glue. |
| 2026-04-27 | Removed the Python package implementation and made `rumba` a top-level Rust/PyO3 extension module. |
| 2026-04-27 | Removed the optional Python example script; remaining Python files are public API tests only. |
