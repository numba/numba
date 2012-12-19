-------------
CUDA Device
-------------

For multi-GPU machines, users may want to select which GPU to use.
By default the CUDA driver selects the fastest GPU as the device 0,
which is the default device used by NumbaPro.

The features introduced on this page is optional for most users.

Device Selection
----------------

Device selection must be done before any cuda feature is used.

::
    from numbapro import cuda
    cuda.select_device(0)
    
This creates a new CUDA context with the selected device.  
The context is associated with the current thread.
NumbaPro currently allows only one context per thread.

The device can be closed by:

::
    cuda.close()
    
This releases the CUDA context from the current thread.

Users can than create a new context with another device.

::
    cuda.select_device(1)  # assuming we have 2 GPUs
    

Multi-Device
-------------

It is possible to use multiple devices by using multiple threads and 
associating different devices to different threads.  More...