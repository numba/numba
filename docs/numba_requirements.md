# Requirements on Numba as needed by NumbaPro #

1. The pipeline MUST be able to accept a python function and emits a LLVM module.
2. The LLVM module MUST contain a definition of the function being compiled.
3. Symbols which type cannot be determined MUST be represented as the monotype.
4. The pipeline MUST NOT require type information of any global symbol.
5. When the type information of a symbol is given, the pipeline SHOULD 
specialize all uses of the symbol.
6. Common math and utility functions MUST be compiled into external LLVM
function declarations that are defined by the runtime API.
7. If a symbol is not locally defined in the module and the symbol is not 
defined by the runtime API, it MUST be referring to a Numba/NumbaPro compiled 
function.
8. The pipeline MUST NOT assume any machine-dependent capability, 
including but not limited to:
    - array representation,
    - exception handling,
    - SIMD vector support,
    - runtime stack operations (e.g. unwind, getting/setting the stack pointer),
    - calling convention.
    - implementation of the runtime API.
    

## Notes ##

Requirements 3-5 suggest that the type inferer must be able to:

- partially/fully type an untyped function;
- partially/fully type a partially typed function given additional type information.


This should allow the pipeline to support cases like:

```python

@autojit
def foo(x):
    if x > 1:
        return bar(x)
    else:
        return x
    
@autojit
def bar(x):
    return foo(x - 1)
    
foo(123)
```

Currently, Numba will recursively invoke the compilation of `bar` which triggers
the compilation of `foo` and halts with a maximum recusion depth error.
With the above requirements, `bar` will be untyped in the `foo` compilation 
module and `foo` will be untyped in the `bar` compilation module.
The pipeline should not recursively compile undefined functions.
The symbol resolution can occur at link time.
Optionally, there can be a link-time optimization that types `bar` and `foo` at
their use site.
We can consider inter-function type inference.
If the input to the type inferer is a list of typing constrains,
it should be applicable to a single function, multiple functions or an entire
module.

An other example:

```python
@autojit(target='gpu')
def foo(x):
    dostuffwith(A, x)
    
A = numpy.random.random(100)
```

To fully take advantage of the fast CUDA constant memory, the user should be 
able to redefine `A`; or, create different versions of `foo` by binding to a
different definition of `A`:

```python
foo1 = foo.bind_const(A=numpy.arange(10))
```

The above requirements would allow redefinition of `A` with a different type.

## Terms ##

- *Global symbols*

    All LLVM module level symbols, including global variables, 
    global constants and functions.

- *Monotype*

    The unknown type.  An opaque pointer that points to a PyObject structure.

