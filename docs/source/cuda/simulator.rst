
.. _simulator:

=================================================
Debugging CUDA Python with the the CUDA Simulator
=================================================

Numba includes a CUDA Simulator that implements most of the semantics in CUDA
Python using the Python interpreter and some additional Python code. This can
be used to debug CUDA Python code, either by adding print statements to your
code, or by using the debugger to step through the execution of an individual
thread.

Execution of kernels is performed by the simulator one block at a time. One
thread is spawned for each thread in the block, and scheduling of the execution
of these threads is left up to the operating system.

Using the simulator
===================

The simulator is enabled by setting the environment variable
:envvar:`NUMBA_ENABLE_CUDASIM` to 1. CUDA Python code may then be executed as
normal. The easiest way to use the debugger inside a kernel is to only stop a
single thread, otherwise the interaction with the debugger is difficult to
handle. For example, the kernel below will  stop in the thread ``<<<(3,0,0), (1,
0, 0)>>>``::

    @cuda.jit
    def vec_add(A, B, out):
        x = cuda.threadIdx.x
        bx = cuda.blockIdx.x
        bdx = cuda.blockDim.x
        if x == 1 and bx == 3:
            from pdb import set_trace; set_trace()
        i = bx * bdx + x
        out[i] = A[i] + B[i]

when invoked with a one-dimensional grid and one-dimensional blocks.

Supported features
==================

The simulator aims to provide as complete a simulation of execution on a real
GPU as possible - in particular, the following are supported:

* Atomic operations
* Constant memory
* Local memory
* Shared memory: declarations of shared memory arrays must be on separate source
  lines, since the simulator uses source line information to keep track of
  allocations of shared memory across threads.
* :func:`.syncthreads` is supported - however, in the case where divergent
  threads enter different :func:`.syncthreads` calls, the launch will not fail,
  but unexpected behaviour will occur. A future version of the simulator may
  detect this condition.
* The stream API is supported, but all operations occur sequentially and
  synchronously, unlike on a real device. Synchronising on a stream is therefore
  a no-op.
* The event API is also supported, but provides no meaningful timing
  information.
* Data transfer to and from the GPU - in particular, creating array objects with
  :func:`.device_array` and :func:`.device_array_like`. The APIs for pinned memory
  :func:`.pinned` and :func:`.pinned_array` are also supported, but no pinning
  takes place.
* The driver API implementation of the list of GPU contexts (``cuda.gpus`` and
  ``cuda.cudadrv.devices.gpus``) is supported, and reports a single GPU context.
  This context can be closed and reset as the real one would.
* The :func:`.detect` function is supported, and reports one device called
  `SIMULATOR`.

Some limitations of the simulator include:

* It does not perform type checking/type inference. If any argument types to a
  jitted function are incorrect, or if the specification of the type of any
  local variables are incorrect, this will not be detected by the simulator.
* Only one GPU is simulated.
* Multithreaded accesses to a single GPU are not supported, and will result in
  unexpected behaviour.
* Most of the driver API is unimplemented.
* It is not possible to link PTX code with CUDA Python functions.

Obviously, the speed of the simulator is also much lower than that of a real
device. It may be necessary to reduce the size of input data and the size of the
CUDA grid in order to make debugging with the simulator tractable.
