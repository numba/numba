
==================
Environment Object
==================

The Environment object (Env) is used to maintain references to python objects
that are needed to support compiled functions for both object-mode and
nopython-mode.

In nopython-mode, the Env is used for:

* Storing pyobjects for reconstruction from native values,
  such as:
  * for printing native values of NumPy arrays;
  * for returning or yielding native values back to the interpreter.

In object-mode, the Env is used for:

* storing constant values referenced in the code.
* storing a reference to the function's global dictionary to load global
  values.


The Implementation
==================

The Env is implemented in two parts.  In ``_dynfunc.c``, the Env is defined
as ``EnvironmentObject`` as a Python C-extension type.  In ``lowering.py``,
the `EnvironmentObject`` (exported as ``_dynfunc.Environment``) is extended
to support necessary operations needed at lowering.


Serialization
-------------

The Env supports being pickled.  Compilation cache files and ahead-of-time
compiled modules serialize all the used Envs for recreation at the runtime.

Usage
-----

At the start of the lowering for a function or a generator, an Env is created.
Throughout the compilation, the Env is mutated to attach additional
information.  The compiled code references an Env via a global variable in
the emitted LLVM IR.  The global variable is zero-initialized with "common"
linkage, which is the default linkage for C global values.  The use of this
linkage allows multiple definitions of the global variable to be merged into
a single definition when the modules are linked together.  The name of the
global variable is computed from the name of the function
(see ``FunctionDescriptor.env_name`` and ``.get_env_name()`` of the target
context).

The Env is initialized when the compiled-function is loaded. The JIT engine
finds the address of the associated global variable for the Env and stores the
address of the Env into it.  For cached functions, the same process applies.
For ahead-of-time compiled functions, the module initializer in the generated
library is responsible for initializing the global variables of all the Envs
in the module.
