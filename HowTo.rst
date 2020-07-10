========
Features
========

DPPL is currently implemented using OpenCL 2.1. The features currently available
are listed below with the help of sample code snippets. In this release we have
the implementation of the OAK approach described in MS138 in section 4.3.2. The
new decorator is described below.

To access the features driver module have to be imported from numba.dppl.dppl_driver

New Decorator
=============

The new decorator included in this release is *dppl.kernel*. Currently this decorator
takes only one option *access_types* which is explained below with the help of an example.
Users can write OpenCL tpye kernels where they can identify the global id of the work item
being executed. The supported methods inside a decorated function are:

- dppl.get_global_id(dimidx)
- dppl.get_local_id(dimidx)
- dppl.get_group_num(dimidx)
- dppl.get_num_groups(dimidx)
- dppl.get_work_dim()
- dppl.get_global_size(dimidx)
- dppl.get_local_size(dimidx)

Currently no support is provided for local memory in the device and everything is in the
global memory. Barrier and other memory fences will be provided once support for local
memory is completed.


Device Environment
==================

To invoke a kernel a device environemnt is required. The device environment can be
initialized by the following methods:

- driver.runtime.get_gpu_device()
- driver.runtime.get_cpu_device()


Device Array
============

Device arrays are used for representing memory buffers in the device. Device Array
supports only ndarrays in this release. Convenience
methods are provided to allocate a memory buffer represnting ndarrays in the device.
They are:

- device_env.copy_array_to_device(ndarray)              :   Allocate buffer of size ndarray
                                                            and copy the data from host to
                                                            device.

- driver.DeviceArray(device_env.get_env_ptr(), ndarray) :   Allocate buffer of size ndarray.


Primitive types are passed by value to the kernel, currently supported are int, float, double.


Math Kernels
============

This release has support for math kernels. See numba/dppl/tests/dppl/test_math_functions.py
for more details.


========
Examples
========

Sum of two 1d arrays
====================

Full example can be found at numba/dppl/examples/sum.py.

To write a program that sums two 1d arrays we at first need a OpenCL device environment.
We can get the environment by using *ocldrv.runtime.get_gpu_device()* for getting the
GPU environment or *ocldrv.runtime.get_cpu_device(data)* for the CPU environment. We then
need to copy the data (which has to be an ndarray) to the device (CPU or GPU) through OpenCL,
where *device_env.copy_array_to_device(data)* will read the ndarray and copy that to the device
and *ocldrv.DeviceArray(device_env.get_env_ptr(), data)* will create a buffer in the device
that has the same memory size as the ndarray being passed. The OpenCL Kernel in the
folllowing example is *data_parallel_sum*. To get the id of the work item we are currently
executing we need to use the  *dppl.get_global_id(0)*, since this example only 1 dimension
we only need to get the id in dimension 0.

While invoking the kernel we need to pass the device environment and the global work size.
After the kernel is executed we want to get the data that contains the sum of the two 1d arrays
back to the host and we can use *device_env.copy_array_from_device(ddata)*.

.. code-block:: python

    @dppl.kernel
    def data_parallel_sum(a, b, c):
        i = dppl.get_global_id(0)
        c[i] = a[i] + b[i]

    global_size = 10
    N = global_size

    a = np.array(np.random.random(N), dtype=np.float32)
    b = np.array(np.random.random(N), dtype=np.float32)
    c = np.ones_like(a)

    # Select a device for executing the kernel
    device_env = None
    try:
        device_env = ocldrv.runtime.get_gpu_device()
    except:
         try:
            device_env = ocldrv.runtime.get_cpu_device()
    except:
        raise SystemExit()

    # Copy the data to the device
    dA = device_env.copy_array_to_device(a)
    dB = device_env.copy_array_to_device(b)
    dC = ocldrv.DeviceArray(device_env.get_env_ptr(), c)

    data_parallel_sum[device_env, global_size](dA, dB, dC)
    device_env.copy_array_from_device(dC)

ndArray Support
===============

Support for passing ndarray directly to kernels is also supported.

Full example can be found at numba/dppl/examples/sum_ndarray.py

For availing this feature instead of creating device buffers explicitly like the previous
example, users can directly pass the ndarray to the kernel. Internally it will result in
copying the existing data in the ndarray to the device and will copy it back after the kernel
is done executing.

In the previous example we can see some redundant work being done. The buffer
that will hold the result of the summation in the device does not need to be copied from the host
and the input data which will be added does not need to be copied back to the host after the
kernel has executed. To reduce doing redundant work, users can provide hints to the compiler
using the access_types to the function decorator. Currently, there are three access types:
*read_only* meaning data will only be copied from host to device, *write_only* meaning memory
will be allocated in device and will be copied back to host and *read_write* which will both
copy data to and from device.


Reduction
=========

This example will demonstrate a sum reduction of 1d array.

Full example can be found at numba/dppl/examples/sum_reduction.py.

In this example to sum the 1d array we invoke the Kernel multiple times.
This can be implemented by invoking the kernel once, but that requires
support for local device memory and barrier, which is a work in progress.


==============
ParFor Support
==============

*Parallel For* is supported in this release for upto 3 dimensions.

Full examples can be found in numba/dppl/examples/pa_examples/


=======
Testing
=======

All examples can be found in numba/dppl/examples/

All tests can be found in numba/dppl/tests/dppl and can be triggered by the following command:

``python -m numba.runtests numba.dppl.tests``
