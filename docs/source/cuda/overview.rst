Overview
========

Numba supports CUDA GPU programming by directly compiling a restricted subset
of python code into CUDA kernels and device functions following the CUDA
execution model.  Numba supports NumPy arrays.  Kernels written in Numba
appears to have direct access to NumPy arrays.  NumPy array is transferred
between the CPU and the GPU automatically.


Terminologies
-------------

A several important terminologies in the topic of CUDA programming is listed
here:

- *host*: the CPU
- *device*: the GPU
- *host memory*: the system main memory
- *device memory*: onboard memory on a GPU card
- *kernels*: a GPU function launched by the host and executed on the device
- *device function*: a GPU function executed on the device and can only be
called from the device

Supported GPUs
--------------

Numba supports CUDA-enabled GPUS with compute capability 2.0 or above with an
up-to-data Nvidia driver.



Missing CUDA Features
---------------------

Numba does not implement all features of CUDA, yet.  Some missing features
are listed below:

* dynamic parallelism
* texture memory


