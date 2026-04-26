# Rumba

Rumba is an experimental, independent implementation of a very small
Numba-like API. It does not import or depend on Numba at runtime.

Current development command:

```bash
cd rumba
maturin develop
python -m pytest
```

There are no Python implementation files for the `rumba` package. The only
Python files kept in this tree are public API tests.

The initial execution path is:

```text
Python function -> Rust/PyO3 frontend -> Rumba AST -> typed Rumba AST
  -> generated C -> shared library -> Rust native invocation
```

Only scalar `int64`, `float64`, and `bool` arguments and scalar returns are
supported in this first slice. Rust owns the importable module, decorator API,
dispatcher, C emission, cache metadata, compiler selection, shared-library
compilation, and native invocation.
