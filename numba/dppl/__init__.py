"""
Module to interact with Intel based devices


Extensions to Numba for Intel GPUs introduce two new features into Numba:
    a.  A new backend that has a new decorator called @dppl.kernel that
        exposes an explicit kernel programming interface similar to the
        existing Numba GPU code-generation backends. The @dppl.kernel
        decorator currently implements a subset of OpenCL’s API through
        Numba’s intrinsic functions.

    b.  A new auto-offload optimizer that does automatic detection and
        offloading of data-parallel code sections on to a GPU or other
        OpenCL/SYCL devices. The auto-offload optimizer is enabled using
        Numba’s default @jit decorator.


Explicit Kernel Prgoramming with new Docorators:


@dppl.kernel

    The @dppl.kernel decorator can be used with or without extra arguments.
    Optionally, users can pass the signature of the arguments to the
    decorator. When a signature is provided to the DK decorator the version
    of the OpenCL kernel generated gets specialized for that type signature.

    ---------------------------------------------------------------------------
    @dppl.kernel
    def data_parallel_sum(a, b, c):
        i = dppl.get_global_id(0)
        c[i] = a[i] + b[i]
    ---------------------------------------------------------------------------

    To invoke the above function users will need to provide a
    global size (OpenCL) which is the size of a (same as b and c) and a
    local size (dppl.DEFAULT_LOCAL_SIZE if user don't want to specify).
    Example shown below:

    ---------------------------------------------------------------------------
    data_parallel_sum[len(a), dppl.DEFAULT_LOCAL_SIZE](dA, dB, dC)
    ---------------------------------------------------------------------------


@dppl.func

    The @dppl.func decorator is the other decorator provided in the explicit
    kernel programming model. This decorator allows users to write “device”
    functions that can be invoked from inside DK functions but cannot be invoked
    from the host. The decorator also supports type specialization as with the
    DK decorator. Functions decorated with @dppl.func will also be JIT compiled
    and inlined into the OpenCL Program containing the @dppl.kernel function
    calling it. A @dppl.func will not be launched as an OpenCL kernel.

    ---------------------------------------------------------------------------
    @dppl.func
    def bar(a):
        return a*a

    @dppl.kernel
    def foo(in, out):
        i = dppl.get_global_id(0)
        out[i] = bar(in[i])
    ---------------------------------------------------------------------------


Intrinsic Functions:

    The following table has the list of intrinsic functions that can be directly
    used inside a DK function. All the functions are equivalent to the similarly
    named OpenCL function. Wherever there is an implementation difference
    between the Numba-PyDPPL version and the OpenCL version, the difference is
    explained in table. Note that these functions cannot be used anywhere else
    outside of a DK function in a Numba application. Readers are referred to the
    OpenCL API specs to review the functionality of each function.

    +----------------------+----------------------------+----------------------+
    | Numba-DPPL intrinsic | Equivalent OpenCL function |         Notes        |
    +----------------------+----------------------------+----------------------+
    | get_global_id        | get_global_id              |                      |
    +----------------------+----------------------------+----------------------+
    | get_local_id         | get_local_id               |                      |
    +----------------------+----------------------------+----------------------+
    | get_global_size      | get_global_size            |                      |
    +----------------------+----------------------------+----------------------+
    | get_local_size       | get_local_size             |                      |
    +----------------------+----------------------------+----------------------+
    | get_group_id         | get_group_id               |                      |
    +----------------------+----------------------------+----------------------+
    | get_num_groups       | get_num_groups             |                      |
    +----------------------+----------------------------+----------------------+
    | get_work_dim         | get_work_dim               |                      |
    +----------------------+----------------------------+----------------------+
    | barrier              | barrier                    |                      |
    +----------------------+----------------------------+----------------------+
    | mem_fence            | mem_fence                  |                      |
    +----------------------+----------------------------+----------------------+
    | sub_group_barrier    | sub_group_barrier          | Does not take any    |
    |                      |                            | argument and is      |
    |                      |                            | equivalent to calling|
    |                      |                            | barrier with the     |
    |                      |                            | CLK_LOCAL_MEM_FENCE  |
    |                      |                            | argument.            |
    +----------------------+----------------------------+----------------------+


Other Intrinsic Functions

    The explicit kernel programming feature provides some additional
    helper/intrinsic functions that do not have a one-to-one mapping with OpenCL
    API functions. The following table has the list of all such currently
    supported functions. As with the other intrinsic functions, these functions
    can only be used inside a DK decorated function.


    +------------------+-------------------------------+-------------------------+
    |    Intrinsic     |         Signature             |         Notes           |
    +------------------+-------------------------------+-------------------------+
    |print             |print(varargs)                 |The print function is a  |
    |                  |                               |subset of the OpenCL     |
    |                  |                               |printf function. The     |
    |                  |                               |Numba-DPPL version of    |
    |                  |                               |print supports only int, |
    |                  |                               |string, and float        |
    |                  |                               |arguments.               |
    +------------------+-------------------------------+-------------------------+
    |local.static_alloc|local.static_alloc(shape,dtype)|This function allow users|
    |                  |                               |to create local memory   |
    |                  |                               |that's only accessible to|
    |                  |                               |work items in a workgroup|
    |                  |                               |                         |
    |                  |                               |Required Arguments:      |
    |                  |                               |shape: An integer or a   |
    |                  |                               |       tuple of integers |
    |                  |                               |dtype: Integer, float or |
    |                  |                               |       Numba supported   |
    |                  |                               |       NumPy dtypes      |
    +------------------+-------------------------------+-------------------------+
    |atomic.add        |atomic.add(addr, value)        |The atomic.add function  |
    |                  |                               |performs an atomicrmw    |
    |                  |                               |(read-modify-write       |
    |                  |                               |operation) on the        |
    |                  |                               |operand “addr” using the |
    |                  |                               |operand “value”.         |
    |                  |                               |                         |
    |                  |                               |Note that the atomic.add |
    |                  |                               |operation only supports  |
    |                  |                               |integer data types.      |
    +------------------+-------------------------------+-------------------------+
    |atomic.sub        |atomic.sub(addr, value)        |Same as atomic.add but   |
    |                  |                               |does subtraction instead |
    |                  |                               |of addition.             |
    |                  |                               |                         |
    |                  |                               |Note that the atomic.add |
    |                  |                               |operation only supports  |
    |                  |                               |integer data types.      |
    +-----------------+-------------------------------+--------------------------+


Complete Example using @dppl.kernel:

    ---------------------------------------------------------------------------
    import numpy as np
    from numba import dppl
    import dppl.ocldrv as ocldrv

    @dppl.kernel
    def data_parallel_sum(a, b, c):
        i = dppl.get_global_id(0)
        c[i] = a[i] + b[i]

    def driver(device_env, a, b, c, global_size):
        # Copy the data to the device
        dA = device_env.copy_array_to_device(a)
        dB = device_env.copy_array_to_device(b)
        dC = device_env.create_device_array(c)

        print("before : ", dA._ndarray)
        print("before : ", dB._ndarray)
        print("before : ", dC._ndarray)
        data_parallel_sum[global_size, dppl.DEFAULT_LOCAL_SIZE](dA, dB, dC)
        device_env.copy_array_from_device(dC)
        print("after : ", dC._ndarray)

    def main():
        global_size = 10
        N = global_size
        print("N", N)

        a = np.array(np.random.random(N), dtype=np.float32)
        b = np.array(np.random.random(N), dtype=np.float32)
        c = np.ones_like(a)

        if ocldrv.has_gpu_device:
            with ocldrv.igpu_context(0) as device_env:
                driver(device_env, a, b, c, global_size)
        elif ocldrv.has_cpu_device:
            with ocldrv.cpu_context(0) as device_env:
                driver(device_env, a, b, c, global_size)
        else:
            print("No device found")
            exit()

        print("Done...")

    if __name__ == '__main__':
        main()
    ---------------------------------------------------------------------------


"""

from __future__ import print_function, absolute_import, division

from numba import config
import numba.testing

from numba.dppl_config import *
if dppl_present:
    from .device_init import *
else:
    raise ImportError("Importing dppl failed")

def test(*args, **kwargs):
    if not dppl_present and not is_available():
        dppl_error()

    return numba.testing.test("numba.dppl.tests", *args, **kwargs)
