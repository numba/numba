
Numba Project Directory Layout
==============================

::

  Numba project root
    ├── benchmarks
    ├── bin
    ├── docs
    │   ├── source : Documentation source root
    │   ├── cuda : CUDA GPU support documentation
    │   ├── cuda-reference : CUDA support reference material
    │   ├── developer : Resources for anyone interested in making contributions to the Numba codebase.
    │   ├── extending : Resources for extending Numba to work with userspace types and functions.
    │   ├── hsa : HSA APU support documentation
    │   ├── proposals : Repository for formal proposal writeups.
    │   ├── reference : Reference material and miscellaneous information for Numba usage.
    │   └── user :
    ├── examples : Use examples
    ├── numba : Package root
    │   ├── annotations :
    │   ├── cuda : CUDA support subpackage
    │   │   ├── cudadrv :
    │   │   ├── kernels :
    │   │   ├── simulator :
    │   │   │   └── cudadrv :
    │   │   └── tests : CUDA specific unit tests
    │   │       ├── cudadrv :
    │   │       │   └── data :
    │   │       ├── cudapy :
    │   │       ├── cudasim :
    │   │       └── nocuda :
    │   ├── datamodel : Models converting objects between high level types and LLVM low level types.
    │   │   ├── manager.py :
    │   │   ├── models.py :
    │   │   ├── packer.py :
    │   │   ├── registry.py :
    │   │   └── testing.py :
    │   ├── hsa :
    │   │   ├── hlc :
    │   │   ├── hsadrv :
    │   │   └── tests :
    │   │       ├── hsadrv :
    │   │       └── hsapy :
    │   ├── jitclass :
    │   ├── npyufunc :
    │   ├── pycc :
    │   ├── rewrites :
    │   ├── runtime :
    │   ├── scripts :
    │   ├── servicelib :
    │   ├── targets :
    │   ├── testing :
    │   ├── tests :
    │   │   ├── npyufunc :
    │   │   └── pycc_distutils_usecase :
    │   ├── typeconv :
    │   ├── types : Numba internal type implementations
    │   │   ├── abstract.py :
    │   │   ├── common.py :
    │   │   ├── containers.py :
    │   │   ├── functions.py :
    │   │   ├── iterators.py :
    │   │   ├── misc.py :
    │   │   ├── npytypes.py :
    │   │   └── scalars.py :
    │   ├── typing :
    │   └── unsafe :
    └── tutorials : Tutorials demonstrating Numba use.