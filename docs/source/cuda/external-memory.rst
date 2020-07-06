.. _cuda-emm-plugin:

=================================================
External Memory Management (EMM) Plugin interface
=================================================

The :ref:`CUDA Array Interface <cuda-array-interface>` enables sharing of data
between different Python libraries that access CUDA devices. However, each
library manages its own memory distinctly from the others. For example:

- By default, Numba allocates memory on CUDA devices by interacting with the
  CUDA driver API to call functions such as ``cuMemAlloc`` and ``cuMemFree``,
  which is suitable for many use cases.
- The RAPIDS libraries (cuDF, cuML, etc.) use the `RAPIDS Memory Manager (RMM)
  <https://github.com/rapidsai/rmm>`_ for allocating device memory.
- `CuPy <https://cupy.chainer.org/>`_ includes a `memory pool implementation
  <https://docs-cupy.chainer.org/en/stable/reference/memory.html>`_ for both
  device and pinned memory.

When multiple CUDA-aware libraries are used together, it may be preferable for
Numba to defer to another library for memory management. The EMM Plugin
interface facilitates this, by enabling Numba to use another CUDA-aware library
for all allocations and deallocations.

An EMM Plugin is used to facilitate the use of an external library for memory
management. An EMM Plugin can be a part of an external library, or could be
implemented as a separate library.


Overview of External Memory Management
======================================

When an EMM Plugin is in use (see :ref:`setting-emm-plugin`), Numba will make
memory allocations and deallocations through the Plugin. It will never directly call
functions such as ``cuMemAlloc``, ``cuMemFree``, etc.

EMM Plugins always take responsibility for the management of device memory.
However, not all CUDA-aware libraries also support managing host memory, so a
facility for Numba to continue the management of host memory whilst ceding
control of device memory to the EMM is provided (see
:ref:`host-only-cuda-memory-manager`).


Effects on Deallocation Strategies
----------------------------------

Numba's internal :ref:`deallocation-behavior` is designed to increase efficiency
by deferring deallocations until a significant quantity are pending. It also
provides a mechanism for preventing deallocations entirely during critical
sections, using the :func:`~numba.cuda.defer_cleanup` context manager.

When an EMM Plugin is in use, the deallocation strategy is implemented by the
EMM, and Numba's internal deallocation mechanism is not used. The EMM
Plugin could implement:
  
- A similar strategy to the Numba deallocation behaviour, or
- Something more appropriate to the plugin - for example, deallocated memory
  might immediately be returned to a memory pool.

The ``defer_cleanup`` context manager may behave differently with an EMM Plugin
- an EMM Plugin should be accompanied by documentation of the behaviour of the
``defer_cleanup`` context manager when it is in use. For example, a pool
allocator could always immediately return memory to a pool even when the
context manager is in use, but could choose not to free empty pools until
``defer_cleanup`` is not in use.


Management of other objects
---------------------------

In addition to memory, Numba manages the allocation and deallocation of
:ref:`events <events>`, :ref:`streams <streams>`, and modules (a module is a
compiled object, which is generated from ``@cuda.jit``\ -ted functions). The
management of events, streams, and modules is unchanged by the use of an EMM
Plugin.


Asynchronous allocation and deallocation
----------------------------------------

The present EMM Plugin interface does not provide support for asynchronous
allocation and deallocation. This may be added to a future version of the
interface.


Implementing an EMM Plugin
==========================

An EMM Plugin is implemented by deriving from
:class:`~numba.cuda.BaseCUDAMemoryManager`. A summary of considerations for the
implementation follows:

- Numba instantiates one instance of the EMM Plugin class per context. The
  context that owns an EMM Plugin object is accessible through ``self.context``,
  if required.
- The EMM Plugin is transparent to any code that uses Numba - all its methods
  are invoked by Numba, and never need to be called by code that uses Numba.
- The allocation methods ``memalloc``, ``memhostalloc``, and ``mempin``, should
  use the underlying library to allocate and/or pin device or host memory, and
  construct an instance of a :ref:`memory pointer <memory-pointers>`
  representing the memory to return back to Numba. These methods are always
  called when the current CUDA context is the context that owns the EMM Plugin
  instance.
- The ``initialize`` method is called by Numba prior to the first use of the EMM
  Plugin object for a context. This method should do anything required to
  prepare the underlying library for allocations in the current context. This
  method may be called multiple times, and must not invalidate previous state
  when it is called.
- The ``reset`` method is called when all allocations in the context are to be
  cleaned up. It may be called even prior to ``initialize``, and an EMM Plugin
  implementation needs to guard against this.
- To support inter-GPU communication, the ``get_ipc_handle`` method should
  provide an :class:`~numba.cuda.IpcHandle` for a given
  :class:`~numba.cuda.MemoryPointer` instance. This method is part of the EMM
  interface (rather than being handled within Numba) because the base address of
  the allocation is only known by the underlying library. Closing an IPC handle
  is handled internally within Numba.
- It is optional to provide memory info from the ``get_memory_info`` method, which
  provides a count of the total and free memory on the device for the context.
  It is preferrable to implement the method, but this may not be practical for
  all allocators. If memory info is not provided, this method should raise a
  :class:`RuntimeError`.
- The ``defer_cleanup`` method should return a context manager that ensures that
  expensive cleanup operations are avoided whilst it is active. The nuances of
  this will vary between plugins, so the plugin documentation should include an
  explanation of how deferring cleanup affects deallocations, and performance in
  general.
- The ``interface_version`` property is used to ensure that the plugin version
  matches the interface provided by the version of Numba. At present, this
  should always be 1.

Full documentation for the base class follows:

.. autoclass:: numba.cuda.BaseCUDAMemoryManager
   :members: memalloc, memhostalloc, mempin, initialize, get_ipc_handle,
             get_memory_info, reset, defer_cleanup, interface_version
   :member-order: bysource


.. _host-only-cuda-memory-manager:

The Host-Only CUDA Memory Manager
---------------------------------

Some external memory managers will support management of on-device memory but
not host memory. For implementing EMM Plugins using one of these memory
managers, a partial implementation of a plugin that implements host-side
allocation and pinning is provided. To use it, derive from
:class:`~numba.cuda.HostOnlyCUDAMemoryManager` instead of
:class:`~numba.cuda.BaseCUDAMemoryManager`. Guidelines for using this class
are:

- The host-only memory manager implements ``memhostalloc`` and ``mempin`` - the
  EMM Plugin should still implement ``memalloc``.
- If ``reset`` is overridden, it must also call ``super().reset()`` to allow the
  host allocations to be cleaned up.
- If ``defer_cleanup`` is overridden, it must hold an active context manager
  from ``super().defer_cleanup()`` to ensure that host-side cleanup is also
  deferred.

Documentation for the methods of :class:`~numba.cuda.HostOnlyCUDAMemoryManager`
follows:

.. autoclass:: numba.cuda.HostOnlyCUDAMemoryManager
   :members: memhostalloc, mempin, reset, defer_cleanup
   :member-order: bysource


Classes and structures of returned objects
==========================================

This section provides an overview of the classes and structures that need to be
constructed by an EMM Plugin.

.. _memory-pointers:

Memory Pointers
---------------

EMM Plugins should construct memory pointer instances that represent their
allocations, for return to Numba. The appropriate memory pointer class to use in
each method is:

- :class:`~numba.cuda.MemoryPointer`: returned from ``memalloc``
- :class:`~numba.cuda.MappedMemory`: returned from ``memhostalloc`` or
  ``mempin`` when the host memory is mapped into the device memory space.
- :class:`~numba.cuda.PinnedMemory`: return from ``memhostalloc`` or ``mempin``
  when the host memory is not mapped into the device memory space.

Memory pointers can take a finalizer, which is a function that is called when
the buffer is no longer needed. Usually the finalizer will make a call to the
memory management library (either internal to Numba, or external if allocated
by an EMM Plugin) to inform it that the memory is no longer required, and that
it could potentially be freed and/or unpinned. The memory manager may choose to
defer actually cleaning up the memory to any later time after the finalizer
runs - it is not required to free the buffer immediately.

Documentation for the memory pointer classes follows.

.. autoclass:: numba.cuda.MemoryPointer

The ``AutoFreePointer`` class need not be used directly, but is documented here
as it is subclassed by :class:`numba.cuda.MappedMemory`:

.. autoclass:: numba.cuda.cudadrv.driver.AutoFreePointer

.. autoclass:: numba.cuda.MappedMemory

.. autoclass:: numba.cuda.PinnedMemory


Memory Info
-----------

If an implementation of
:meth:`~numba.cuda.BaseCUDAMemoryManager.get_memory_info` is to provide a
result, then it should return an instance of the ``MemoryInfo`` named tuple:

.. autoclass:: numba.cuda.MemoryInfo


IPC
---

An instance of ``IpcHandle`` is required to be returned from an implementation
of :meth:`~numba.cuda.BaseCUDAMemoryManager.get_ipc_handle`:

.. autoclass:: numba.cuda.IpcHandle

Guidance for constructing an IPC handle in the context of implementing an EMM
Plugin:

- The ``memory`` parameter passed to the ``get_ipc_handle`` method of an EMM
  Plugin can be passed as the ``base`` parameter.
- A suitable type for the ``handle`` can be constructed as ``ctypes.c_byte *
  64``. The data for ``handle`` must be populated using a method for obtaining a
  CUDA IPC handle appropriate to the underlying library.
- ``size`` should match the size of the original allocation, which can be
  obtained with ``memory.size`` in ``get_ipc_handle``.
- An appropriate value for ``source_info`` can be created by calling
  ``self.context.device.get_device_identity()``.
- If the underlying memory does not point to the base of an allocation returned
  by the CUDA driver or runtime API (e.g. if a pool allocator is in use) then
  the ``offset`` from the base must be provided.


.. _setting-emm-plugin:

Setting the EMM Plugin
======================

By default, Numba uses its internal memory management - if an EMM Plugin is to
be used, it must be configured. There are two mechanisms for configuring the use
of an EMM Plugin: an environment variable, and a function.


Environment variable
--------------------

A module name can be provided in the environment variable,
``NUMBA_CUDA_MEMORY_MANAGER``. If this environment variable is set, Numba will
attempt to import the module, and and use its ``_numba_memory_manager`` global
variable as the memory manager class. This is primarily useful for running the
Numba test suite with an EMM Plugin, e.g.:

.. code::

   $ NUMBA_CUDA_MEMORY_MANAGER=rmm python -m numba.runtests numba.cuda.tests


Function
--------

The :func:`~numba.cuda.set_memory_manager` function can be used to set the
memory manager at runtime. This must be called prior to the initialization of
any contexts, and EMM Plugin instances are instantiated along with contexts.

It is recommended that the memory manager is set once prior to using any CUDA
functionality, and left unchanged for the remainder of execution.

.. autofunction:: numba.cuda.set_memory_manager
