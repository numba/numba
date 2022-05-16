Types on CUDA target
====================

Note: This page is about types specific to CUDA targets. Most types on CPU targets are also
available in CUDA target, for information on these types, see :ref:`numba-types`.

Vector Types
~~~~~~~~~~~~

`CUDA Vector Types <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#built-in-vector-types>`_
are available in the kernel. There are 2 important distinctions from vector types in numba CUDA:

First, the recommended names for vector types in Numba CUDA is formatted as ``base_typexN``,
where `base_type` is the base type of the vector, and `N` is the number of elements in the vector.
Examples include ``int64x3``, ``uint16x4``, ``float32x4``, etc. This is consistent in type names with
the pydata community where bit widths are explicit. For new Numba CUDA kernels, this is the recommended
way to write vector types.

For convenience, users who prefer to adapt existing kernels from native CUDA to Numba CUDA may use
aliases consistent with its native namings. For example, ``float3`` aliases ``float32x3``,
``long3`` aliases ``int32x3`` or ``int64x3``, depending on platform specification, etc. 

Second, unlike native CUDA where factory functions are used, vector types are constructed directly
with their constructor. For example, to construct a ``float32x3``:

.. code-block:: python3

    from numba.cuda import float32x3

    # In kernel
    f3 = float32x3(float32(0.0), float32(-1.0), float32(1.0))

Additionally, vector types can be constructed from a combination of vector and
primitive types, as long as the total number of components matches the result
vector type. For example, all of the following constructions are valid:

.. code-block:: python3

    zero = uint32(0)
    u2 = uint32x2(1, 2)
    # Construct a 3-component vector with primitive type and a 2-component vector
    u3 = uint32x3(zero, u2)
    # Construct a 4-component vector with 2 2-component vectors
    u4 = uint32x4(u2, u2)

The 1st, 2nd, 3rd and 4th component of the vector type can be accessed through fields 
``x``, ``y``, ``z``, and ``w`` respectively. The components are immutable after
construction in the present version of Numba; it is expected that support for
mutating vector components will be added in a future release.