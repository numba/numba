.. _nbep-7:

===============================================
NBEP 7: CUDA External Memory Management Plugins
===============================================

:Author: Graham Markall, NVIDIA
:Contributors: Thomson Comer, Peter Entschev, Leo Fang, John Kirkham, Keith Kraus
:Date: March 2020
:Status: Final

Background and goals
--------------------

The :ref:`CUDA Array Interface <cuda-array-interface>` enables sharing of data
between different Python libraries that access CUDA devices. However, each
library manages its own memory distinctly from the others. For example:


* `Numba <https://numba.pydata.org/>`_ internally manages memory for the creation
  of device and mapped host arrays.
* `The RAPIDS libraries <https://rapids.ai/>`_ (cuDF, cuML, etc.) use the `Rapids
  Memory Manager <https://github.com/rapidsai/rmm>`_ for allocating device
  memory.
* `CuPy <https://cupy.chainer.org/>`_ includes a `memory pool
  implementation <https://docs-cupy.chainer.org/en/stable/reference/memory.html>`_
  for both device and pinned memory.

The goal of this NBEP is to describe a plugin interface that enables Numba's
internal memory management to be replaced with an external memory manager by the
user. When the plugin interface is in use, Numba no longer directly allocates or
frees any memory when creating arrays, but instead requests allocations and
frees through the external manager.

Requirements
------------

Provide an *External Memory Manager (EMM)* interface in Numba.


* When the EMM is in use, Numba will make all memory allocation using the EMM.
  It will never directly call functions such as ``CuMemAlloc``\ , ``cuMemFree``\ , etc.
* When not using an *External Memory Manager (EMM)*\ , Numba's present behaviour
  is unchanged (at the time of writing, the current version is the 0.48
  release).

If an EMM is to be used, it will entirely replace Numba's internal memory
management for the duration of program execution. An interface for setting the
memory manager will be provided.

Device vs. Host memory
^^^^^^^^^^^^^^^^^^^^^^^

An EMM will always take responsibility for the management of device memory.
However, not all CUDA memory management libraries also support managing host
memory, so a facility for Numba to continue the management of host memory 
whilst ceding control of device memory to the EMM will be provided.

Deallocation strategies
^^^^^^^^^^^^^^^^^^^^^^^

Numba's internal memory management uses a :ref:`deallocation strategy
<deallocation-behavior>` designed to increase efficiency by deferring
deallocations until a significant quantity are pending. It also provides a
mechanism for preventing deallocations entirely during critical sections, using
the :func:`~numba.cuda.defer_cleanup` context manager.


* When the EMM is not in use, the deallocation strategy and operation of
  ``defer_cleanup`` remain unchanged.
* When the EMM is in use, the deallocation strategy is implemented by the EMM,
  and Numba's internal deallocation mechanism is not used. For example:

  * A similar strategy to Numba's could be implemented by the EMM, or
  * Deallocated memory might immediately be returned to a memory pool.

* The ``defer_cleanup`` context manager may behave differently with an EMM - an
  EMM should be accompanied by documentation of the behaviour of the
  ``defer_cleanup`` context manager when it is in use.

  * For example, a pool allocator could always immediately return memory to a
    pool even when the context manager is in use, but could choose
    not to free empty pools until ``defer_cleanup`` is not in use.

Management of other objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In addition to memory, Numba manages the allocation and deallocation of
:ref:`events <events>`, :ref:`streams <streams>`, and modules (a module is a
compiled object, which is generated from ``@cuda.jit``\ -ted functions). The
management of streams, events, and modules should be unchanged by the presence
or absence of an EMM.

Asynchronous allocation / deallocation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An asynchronous memory manager might provide the facility for an allocation or
free to take a CUDA stream and execute asynchronously. For freeing, this is
unlikely to cause issues since it operates at a layer beneath Python, but for
allocations this could be problematic if the user tries to then launch a kernel
on the default stream from this asynchronous memory allocation.

The interface described in this proposal will not be required to support
asynchronous allocation and deallocation, and as such these use cases will not
be considered further. However, nothing in this proposal should preclude the
straightforward addition of asynchronous operations in future versions of the
interface.

Non-requirements
^^^^^^^^^^^^^^^^

In order to minimise complexity and constrain this proposal to a reasonable
scope, the following will not be supported:


* Using different memory manager implementations for different contexts. All
  contexts will use the same memory manager implementation - either the Numba
  internal implementation or an external implementation.
* Changing the memory manager once execution has begun. It is not practical to
  change the memory manager and retain all allocations. Cleaning up the entire
  state and then changing to a different memory allocator (rather than starting
  a new process) appears to be a rather niche use case.
* Any changes to the ``__cuda_array_interface__`` to further define its semantics,
  e.g. for acquiring / releasing memory as discussed in `Numba Issue
  #4886 <https://github.com/numba/numba/issues/4886>`_ - these are independent,
  and can be addressed as part of separate proposals.
* Managed memory / UVM is not supported. At present Numba does not support UVM -
  see `Numba Issue #4362 <https://github.com/numba/numba/issues/4362>`_ for
  discussion of support.

Interface for Plugin developers
-------------------------------

New classes and functions will be added to ``numba.cuda.cudadrv.driver``:

* ``BaseCUDAMemoryManager`` and ``HostOnlyCUDAMemoryManager``\ : base classes for
  EMM plugin implementations.
* ``set_memory_manager``: a method for registering an external memory manager with
  Numba.

These will be exposed through the public API, in the ``numba.cuda`` module.
Additionally, some classes that are already part of the `driver` module will be
exposed as part of the public API:

* ``MemoryPointer``: used to encapsulate information about a pointer to device
  memory.
* ``MappedMemory``: used to hold information about host memory that is mapped into
  the device address space (a subclass of ``MemoryPointer``\ ).
* ``PinnedMemory``: used to hold information about host memory that is pinned (a
  subclass of ``mviewbuf.MemAlloc``\ , a class internal to Numba).

As an alternative to calling the ``set_memory_manager`` function, an environment
variable can be used to set the memory manager. The value of the environment
variable should be the name of the module containing the memory manager in its
global scope, named ``_numba_memory_manager``\ :

.. code-block::

   export NUMBA_CUDA_MEMORY_MANAGER="<module>"

When this variable is set, Numba will automatically use the memory manager from
the specified module. Calls to ``set_memory_manager`` will issue a warning, but
otherwise be ignored.

Plugin Base Classes
^^^^^^^^^^^^^^^^^^^

An EMM plugin is implemented by inheriting from the ``BaseCUDAMemoryManager``
class, which is defined as:

.. code-block:: python

   class BaseCUDAMemoryManager(object, metaclass=ABCMeta):
       @abstractmethod
       def memalloc(self, size):
           """
           Allocate on-device memory in the current context. Arguments:

           - `size`: Size of allocation in bytes

           Returns: a `MemoryPointer` to the allocated memory.
           """

       @abstractmethod
       def memhostalloc(self, size, mapped, portable, wc):
           """
           Allocate pinned host memory. Arguments:

           - `size`: Size of the allocation in bytes
           - `mapped`: Whether the allocated memory should be mapped into the CUDA
                       address space.
           - `portable`: Whether the memory will be considered pinned by all
                         contexts, and not just the calling context.
           - `wc`: Whether to allocate the memory as write-combined.

           Returns a `MappedMemory` or `PinnedMemory` instance that owns the
           allocated memory, depending on whether the region was mapped into
           device memory.
           """

       @abstractmethod
       def mempin(self, owner, pointer, size, mapped):
           """
           Pin a region of host memory that is already allocated. Arguments:

           - `owner`: An object owning the memory - e.g. a `DeviceNDArray`.
           - `pointer`: The pointer to the beginning of the region to pin.
           - `size`: The size of the region to pin.
           - `mapped`: Whether the region should also be mapped into device memory.

           Returns a `MappedMemory` or `PinnedMemory` instance that refers to the
           allocated memory, depending on whether the region was mapped into device
           memory.
           """

       @abstractmethod
       def initialize(self):
           """
           Perform any initialization required for the EMM plugin to be ready to
           use.
           """

       @abstractmethod
       def get_memory_info(self):
           """
           Returns (free, total) memory in bytes in the context
           """

       @abstractmethod
       def get_ipc_handle(self, memory):
           """
           Return an `IpcHandle` from a GPU allocation. Arguments:

           - `memory`: A `MemoryPointer` for which the IPC handle should be created.
           """

       @abstractmethod
       def reset(self):
           """
           Clear up all memory allocated in this context.
           """

       @abstractmethod
       def defer_cleanup(self):
           """
           Returns a context manager that ensures the implementation of deferred
           cleanup whilst it is active.
           """

       @property
       @abstractmethod
       def interface_version(self):
           """
           Returns an integer specifying the version of the EMM Plugin interface
           supported by the plugin implementation. Should always return 1 for
           implementations described in this proposal.
           """

All of the methods of an EMM plugin are called from within Numba - they never
need to be invoked directly by a Numba user.

The ``initialize`` method is called by Numba prior to any memory allocations
being requested. This gives the EMM an opportunity to initialize any data
structures, etc., that it needs for its normal operations. The method may be
called multiple times during the lifetime of the program - subsequent calls
should not invalidate or reset the state of the EMM.

The ``memalloc``\ , ``memhostalloc``\ , and ``mempin`` methods are called when Numba
requires an allocation of device or host memory, or pinning of host memory.
Device memory should always be allocated in the current context.

``get_ipc_handle`` is called when an IPC handle for an array is required. Note
that there is no method for closing an IPC handle - this is because the
``IpcHandle`` object constructed by ``get_ipc_handle`` contains a ``close()`` method
as part of its definition in Numba, which closes the handle by calling
``cuIpcCloseMemHandle``. It is expected that this is sufficient for general use
cases, so no facility for customising the closing of IPC handles is provided by
the EMM Plugin interface.

``get_memory_info`` may be called at any time after ``initialize``.

``reset`` is called as part of resetting a context. Numba does not normally call
reset spontaneously, but it may be called at the behest of the user. Calls to
``reset`` may even occur before ``initialize`` is called, so the plugin should be
robust against this occurrence.

``defer_cleanup`` is called when the ``numba.cuda.defer_cleanup`` context manager
is used from user code.

``interface_version`` is called by Numba when the memory manager is set, to
ensure that the version of the interface implemented by the plugin is
compatible with the version of Numba in use.

Representing pointers
^^^^^^^^^^^^^^^^^^^^^

Device Memory
~~~~~~~~~~~~~

The ``MemoryPointer`` class is used to represent a pointer to memory. Whilst there
are various details of its implementation, the only aspect relevant to EMM
plugin development is its initialization. The ``__init__`` method has the
following interface:

.. code-block:: python

   class MemoryPointer:
       def __init__(self, context, pointer, size, owner=None, finalizer=None):


* ``context``\ : The context in which the pointer was allocated.
* ``pointer``\ : A ``ctypes`` pointer (e.g. ``ctypes.c_uint64``\ ) holding the address of
  the memory.
* ``size``\ : The size of the allocation in bytes.
* ``owner``\ : The owner is sometimes set by the internals of the class, or used for
  Numba's internal memory management, but need not be provided by the writer of
  an EMM plugin - the default of ``None`` should always suffice.
* ``finalizer``\ : A method that is called when the last reference to the
  ``MemoryPointer`` object is released. Usually this will make a call to the
  external memory management library to inform it that the memory is no longer
  required, and that it could potentially be freed (though the EMM is not
  required to free it immediately).

Host Memory
~~~~~~~~~~~

Memory mapped into the CUDA address space (which is created when the
``memhostalloc`` or ``mempin`` methods are called with ``mapped=True``\ ) is managed
using the ``MappedMemory`` class:

.. code-block:: python

   class MappedMemory(AutoFreePointer):
       def __init__(self, context, pointer, size, owner, finalizer=None):


* ``context``\ : The context in which the pointer was allocated.
* ``pointer``\ : A ``ctypes`` pointer (e.g. ``ctypes.c_void_p``\ ) holding the address of
  the allocated memory.
* ``size``\ : The size of the allocated memory in bytes.
* ``owner``\ : A Python object that owns the memory, e.g. a ``DeviceNDArray``
  instance.
* ``finalizer``\ : A method that is called when the last reference to the
  ``MappedMemory`` object is released. For example, this method could call
  ``cuMemFreeHost`` on the pointer to deallocate the memory immediately.

Note that the inheritance from ``AutoFreePointer`` is an implementation detail and
need not concern the developer of an EMM plugin - ``MemoryPointer`` is higher in
the MRO of ``MappedMemory``.

Memory that is only in the host address space and has been pinned is represented
with the ``PinnedMemory`` class:

.. code-block:: python

   class PinnedMemory(mviewbuf.MemAlloc):
       def __init__(self, context, pointer, size, owner, finalizer=None):


* ``context``\ : The context in which the pointer was allocated.
* ``pointer``\ : A ``ctypes`` pointer (e.g. ``ctypes.c_void_p``\ ) holding the address of
  the pinned memory.
* ``size``\ : The size of the pinned region in bytes.
* ``owner``\ : A Python object that owns the memory, e.g. a ``DeviceNDArray``
  instance.
* ``finalizer``\ : A method that is called when the last reference to the
  ``PinnedMemory`` object is released. This method could e.g. call
  ``cuMemHostUnregister`` on the pointer to unpin the memory immediately.

Providing device memory management only
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some external memory managers will support management of on-device memory but
not host memory. To make it easy to implement an EMM plugin using one of these
managers, Numba will provide a memory manager class with implementations of the
``memhostalloc`` and ``mempin`` methods. An abridged definition of this class
follows:

.. code-block:: python

   class HostOnlyCUDAMemoryManager(BaseCUDAMemoryManager):
       # Unimplemented methods:
       #
       # - memalloc
       # - get_memory_info

       def memhostalloc(self, size, mapped, portable, wc):
           # Implemented.

       def mempin(self, owner, pointer, size, mapped):
           # Implemented.

       def initialize(self):
           # Implemented.
           #
           # Must be called by any subclass when its initialize() method is
           # called.

       def reset(self):
           # Implemented.
           #
           # Must be called by any subclass when its reset() method is
           # called.

       def defer_cleanup(self):
           # Implemented.
           #
           # Must be called by any subclass when its defer_cleanup() method is
           # called.

A class can subclass the ``HostOnlyCUDAMemoryManager`` and then it only needs to
add implementations of methods for on-device memory. Any subclass must observe
the following rules:


* If the subclass implements ``__init__``\ , then it must also call
  ``HostOnlyCUDAMemoryManager.__init__``\ , as this is used to initialize some of
  its data structures (\ ``self.allocations`` and ``self.deallocations``\ ).
* The subclass must implement ``memalloc`` and ``get_memory_info``.
* The ``initialize`` and ``reset`` methods perform initialisation of structures
  used by the ``HostOnlyCUDAMemoryManager``.

  * If the subclass has nothing to do on initialisation (possibly) or reset
    (unlikely) then it need not implement these methods. 
  * However, if it does implement these methods then it must also call the
    methods from ``HostOnlyCUDAMemoryManager`` in its own implementations.

* Similarly if ``defer_cleanup`` is implemented, it should enter the context
  provided by ``HostOnlyCUDAManager.defer_cleanup()`` prior to ``yield``\ ing (or in
  the ``__enter__`` method) and release it prior to exiting (or in the ``__exit__``
  method).

Import order
^^^^^^^^^^^^

The order in which Numba and the library implementing an EMM Plugin should not
matter. For example, if ``rmm`` were to implement and register an EMM Plugin,
then:

.. code-block:: python

   from numba import cuda
   import rmm

and

.. code-block:: python

   import rmm
   from numba import cuda

are equivalent - this is because Numba does not initialize CUDA or allocate any
memory until the first call to a CUDA function - neither instantiating and
registering an EMM plugin, nor importing ``numba.cuda`` causes a call to a CUDA
function.

Numba as a Dependency
^^^^^^^^^^^^^^^^^^^^^

Adding the implementation of an EMM Plugin to a library naturally makes Numba a
dependency of the library where it may not have been previously. In order to
make the dependency optional, if this is desired, one might conditionally
instantiate and register the EMM Plugin like:

.. code-block:: python

   try:
       import numba
       from mylib.numba_utils import MyNumbaMemoryManager
       numba.cuda.cudadrv.driver.set_memory_manager(MyNumbaMemoryManager)
   except:
       print("Numba not importable - not registering EMM Plugin")

so that ``mylib.numba_utils``\ , which contains the implementation of the EMM
Plugin, is only imported if Numba is already present. If Numba is not available,
then ``mylib.numba_utils`` (which necessarily imports ``numba``\ ), will never be
imported.

It is recommended that any library with an EMM Plugin includes at least some
environments with Numba for testing with the EMM Plugin in use, as well as some
environments without Numba, to avoid introducing an accidental Numba dependency.

Example implementation - A RAPIDS Memory Manager (RMM) Plugin
-------------------------------------------------------------

An implementation of an EMM plugin within the `Rapids Memory Manager
(RMM) <https://github.com/rapidsai/rmm>`_ is sketched out in this section. This is
intended to show an overview of the implementation in order to support the
descriptions above and to illustrate how the plugin interface can be used -
different choices may be made for a production-ready implementation.

The plugin implementation consists of additions to `python/rmm/rmm.py
<https://github.com/rapidsai/rmm/blob/d5831ac5ebb5408ee83f63b7c7d03d8870ecb361/python/rmm/rmm.py>`_:

.. code-block:: python

   # New imports:
   from contextlib import context_manager
   # RMM already has Numba as a dependency, so these imports need not be guarded
   # by a check for the presence of numba.
   from numba.cuda import (HostOnlyCUDAMemoryManager, MemoryPointer, IpcHandle,
                           set_memory_manager)


   # New class implementing the EMM Plugin:
   class RMMNumbaManager(HostOnlyCUDAMemoryManager):
       def memalloc(self, size):
           # Allocates device memory using RMM functions. The finalizer for the
           # allocated memory calls back to RMM to free the memory.
           addr = librmm.rmm_alloc(bytesize, 0)
           ctx = cuda.current_context()
           ptr = ctypes.c_uint64(int(addr))
           finalizer = _make_finalizer(addr, stream)
           return MemoryPointer(ctx, ptr, size, finalizer=finalizer)

      def get_ipc_handle(self, memory):
           """ 
           Get an IPC handle for the memory with offset modified by the RMM memory
           pool.
           """
           # This implementation provides a functional implementation and illustrates
           # what get_ipc_handle needs to do, but it is not a very "clean"
           # implementation, and it relies on borrowing bits of Numba internals to
           # initialise ipchandle. 
           #
           # A more polished implementation might make use of additional functions in
           # the RMM C++ layer for initialising IPC handles, and not use any Numba
           # internals.
           ipchandle = (ctypes.c_byte * 64)()  # IPC handle is 64 bytes
           cuda.cudadrv.memory.driver_funcs.cuIpcGetMemHandle(
               ctypes.byref(ipchandle),
               memory.owner.handle,
           )
           source_info = cuda.current_context().device.get_device_identity()
           ptr = memory.device_ctypes_pointer.value
           offset = librmm.rmm_getallocationoffset(ptr, 0)
           return IpcHandle(memory, ipchandle, memory.size, source_info,
                            offset=offset)

       def get_memory_info(self):
           # Returns a tuple of (free, total) using RMM functionality.
           return get_info() # Function defined in rmm.py

       def initialize(self):
           # Nothing required to initialize RMM here, but this method is added
           # to illustrate that the super() method should also be called.
           super().initialize()

       @contextmanager
       def defer_cleanup(self):
           # Does nothing to defer cleanup - a full implementation may choose to
           # implement a different policy.
           with super().defer_cleanup():
               yield

       @property
       def interface_version(self):
           # As required by the specification
           return 1

   # The existing _make_finalizer function is used by RMMNumbaManager:
   def _make_finalizer(handle, stream):
       """
       Factory to make the finalizer function.
       We need to bind *handle* and *stream* into the actual finalizer, which
       takes no args.
       """

       def finalizer():
           """
           Invoked when the MemoryPointer is freed
           """
           librmm.rmm_free(handle, stream)

       return finalizer

   # Utility function register `RMMNumbaManager` as an EMM:
   def use_rmm_for_numba():
       set_memory_manager(RMMNumbaManager)

   # To support `NUMBA_CUDA_MEMORY_MANAGER=rmm`:
   _numba_memory_manager = RMMNumbaManager

Example usage
^^^^^^^^^^^^^

A simple example that configures Numba to use RMM for memory management and
creates a device array is as follows:

.. code-block:: python

   # example.py
   import rmm 
   import numpy as np

   from numba import cuda

   rmm.use_rmm_for_numba()

   a = np.zeros(10)
   d_a = cuda.to_device(a)
   del(d_a)
   print(rmm.csv_log())

Running this should result in output similar to the following:

.. code-block::

   Event Type,Device ID,Address,Stream,Size (bytes),Free Memory,Total Memory,Current Allocs,Start,End,Elapsed,Location
   Alloc,0,0x7fae06600000,0,80,0,0,1,1.10549,1.1074,0.00191666,<path>/numba/numba/cuda/cudadrv/driver.py:683
   Free,0,0x7fae06600000,0,0,0,0,0,1.10798,1.10921,0.00122238,<path>/numba/numba/utils.py:678

Note that there is some scope for improvement in RMM for detecting the line
number at which the allocation / free occurred, but this is outside the scope of
the example in this proposal.

Setting the memory manager through the environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rather than calling ``rmm.use_rmm_for_numba()`` in the example above, the memory
manager could also be set to use RMM globally with an environment variable, so
the Python interpreter is invoked to run the example as:

.. code-block::

   NUMBA_CUDA_MEMORY_MANAGER="rmm.RMMNumbaManager" python example.py

Numba internal changes
----------------------

This section is intended primarily for Numba developers - those with an interest
in the external interface for implementing EMM plugins may choose to skip over
this section.

Current model / implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At present, memory management is implemented in the
:class:`~numba.cuda.cudadrv.driver.Context` class. It maintains lists of
allocations and deallocations:

* ``allocations`` is a ``numba.core.utils.UniqueDict``, created at context
  creation time.
* ``deallocations`` is an instance of the ``_PendingDeallocs`` class, and is created
  when ``Context.prepare_for_use()`` is called.

These are used to track allocations and deallocations of:

* Device memory
* Pinned memory
* Mapped memory
* Streams
* Events
* Modules

The ``_PendingDeallocs`` class implements the deferred deallocation strategy -
cleanup functions (such as ``cuMemFree``\ ) for the items above are added to its
list of pending deallocations by the finalizers of objects representing
allocations. These finalizers are run when the objects owning them are
garbage-collected by the Python interpreter. When the addition of a new
cleanup function to the deallocation list causes the number or size of pending
deallocations to exceed a configured ratio, the ``_PendingDeallocs`` object runs
deallocators for all items it knows about and then clears its internal pending
list.

See :ref:`deallocation-behavior` for more details of this implementation.

Proposed changes
^^^^^^^^^^^^^^^^

This section outlines the major changes that will be made to support the EMM
plugin interface - there will be various small changes to other parts of Numba
that will be required in order to adapt to these changes; an exhaustive list of
these is not provided.

Context changes
~~~~~~~~~~~~~~~

The ``numba.cuda.cudadrv.driver.Context`` class will no longer directly allocate
and free memory. Instead, the context will hold a reference to a memory manager
instance, and its memory allocation methods will call into the memory manager,
e.g.:

.. code-block:: python

       def memalloc(self, size):
           return self.memory_manager.memalloc(size)

       def memhostalloc(self, size, mapped=False, portable=False, wc=False):
           return self.memory_manager.memhostalloc(size, mapped, portable, wc)

       def mempin(self, owner, pointer, size, mapped=False):
           if mapped and not self.device.CAN_MAP_HOST_MEMORY:
               raise CudaDriverError("%s cannot map host memory" % self.device)
           return self.memory_manager.mempin(owner, pointer, size, mapped)

       def prepare_for_use(self):
           self.memory_manager.initialize()

       def get_memory_info(self):
           self.memory_manager.get_memory_info()

       def get_ipc_handle(self, memory):
           return self.memory_manager.get_ipc_handle(memory)

       def reset(self):
           # ... Already-extant reset logic, plus:
           self._memory_manager.reset()

The ``memory_manager`` member is initialised when the context is created.

The ``memunpin`` method (not shown above but currently exists in the ``Context``
class) has never been implemented - it presently raises a ``NotImplementedError``.
This method arguably un-needed - pinned memory is immediately unpinned by its
finalizer, and unpinning before a finalizer runs would invalidate the state of
``PinnedMemory`` objects for which references are still held. It is proposed that
this is removed when making the other changes to the ``Context`` class.

The ``Context`` class will still instantiate ``self.allocations`` and
``self.deallocations`` as before - these will still be used by the context to
manage the allocations and deallocations of events, streams, and modules, which
are not handled by the EMM plugin.

New components of the ``driver`` module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


* ``BaseCUDAMemoryManager``\ : An abstract class, as defined in the plugin interface
  above.
* ``HostOnlyCUDAMemoryManager``\ : A subclass of ``BaseCUDAMemoryManager``\ , with the
  logic from ``Context.memhostalloc`` and ``Context.mempin`` moved into it. This
  class will also create its own ``allocations`` and ``deallocations`` members,
  similarly to how the ``Context`` class creates them. These are used to manage
  the allocations and deallocations of pinned and mapped host memory.
* ``NumbaCUDAMemoryManager``\ : A subclass of ``HostOnlyCUDAMemoryManager``\ , which
  also contains an implementation of ``memalloc`` based on that presently existing
  in the ``Context`` class. This is the default memory manager, and its use
  preserves the behaviour of Numba prior to the addition of the EMM plugin
  interface - that is, all memory allocation and deallocation for Numba arrays
  is handled within Numba.

  * This class shares the ``allocations`` and ``deallocations`` members with its
    parent class ``HostOnlyCUDAMemoryManager``\ , and it uses these for the
    management of device memory that it allocates.

* The ``set_memory_manager`` function, which sets a global pointing to the memory
  manager class. This global initially holds ``NumbaCUDAMemoryManager`` (the
  default).

Staged IPC
~~~~~~~~~~

Staged IPC should not take ownership of the memory that it allocates. When the
default internal memory manager is in use, the memory allocated for the staging
array is already owned. When an EMM plugin is in use, it is not legitimate to
take ownership of the memory.

This change can be made by applying the following small patch, which has been
tested to have no effect on the CUDA test suite:

.. code-block:: diff

   diff --git a/numba/cuda/cudadrv/driver.py b/numba/cuda/cudadrv/driver.py
   index 7832955..f2c1352 100644
   --- a/numba/cuda/cudadrv/driver.py
   +++ b/numba/cuda/cudadrv/driver.py
   @@ -922,7 +922,11 @@ class _StagedIpcImpl(object):
            with cuda.gpus[srcdev.id]:
                impl.close()

   -        return newmem.own()
   +        return newmem

Testing
~~~~~~~

Alongside the addition of appropriate tests for new functionality, there will be
some refactoring of existing tests required, but these changes are not
substantial. Tests of the deallocation strategy (e.g. ``TestDeallocation``\ ,
``TestDeferCleanup``\ ) will need to be modified to ensure that they are
examining the correct set of deallocations. When an EMM plugin is in use, they
will need to be skipped.

Prototyping / experimental implementation
-----------------------------------------

Some prototype / experimental implementations have been produced to guide the
designs presented in this document. The current implementations can be found in:


* Numba branch: https://github.com/gmarkall/numba/tree/grm-numba-nbep-7.
* RMM branch: https://github.com/gmarkall/rmm/tree/grm-numba-nbep-7.
* CuPy implementation:
  https://github.com/gmarkall/nbep-7/blob/master/nbep7/cupy_mempool.py - uses
  an unmodified CuPy.

  * See `CuPy memory management
    docs <https://docs-cupy.chainer.org/en/stable/reference/memory.html>`_.

Current implementation status
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

RMM Plugin
~~~~~~~~~~

For a minimal example, a simple allocation and free using RMM works as expected.
For the example code (similar to the RMM example above):

.. code-block:: python

   import rmm
   import numpy as np

   from numba import cuda

   rmm.use_rmm_for_numba()

   a = np.zeros(10)
   d_a = cuda.to_device(a)
   del(d_a)
   print(rmm.csv_log())

We see the following output:

.. code-block::

   Event Type,Device ID,Address,Stream,Size (bytes),Free Memory,Total Memory,Current Allocs,Start,End,Elapsed,Location
   Alloc,0,0x7f96c7400000,0,80,0,0,1,1.13396,1.13576,0.00180059,<path>/numba/numba/cuda/cudadrv/driver.py:686
   Free,0,0x7f96c7400000,0,0,0,0,0,1.13628,1.13723,0.000956004,<path>/numba/numba/utils.py:678

This output is similar to the expected output from the example usage presented
above (though note that the pointer addresses and timestamps vary compared to
the example), and provides some validation of the example use case.

CuPy Plugin
~~~~~~~~~~~

.. code-block:: python

   from nbep7.cupy_mempool import use_cupy_mm_for_numba
   import numpy as np

   from numba import cuda

   use_cupy_mm_for_numba()

   a = np.zeros(10)
   d_a = cuda.to_device(a)
   del(d_a)

The prototype CuPy plugin has somewhat primitive logging, so we see the output:

.. code-block::

   Allocated 80 bytes at 7f004d400000
   Freeing 80 bytes at 7f004d400000

Numba CUDA Unit tests
^^^^^^^^^^^^^^^^^^^^^

As well as providing correct execution of a simple example, all relevant Numba
CUDA unit tests also pass with the prototype branch, for both the internal memory
manager and the RMM EMM Plugin.

RMM
~~~

The unit test suite can be run with the RMM EMM Plugin with:

.. code-block::

   NUMBA_CUDA_MEMORY_MANAGER=rmm python -m numba.runtests numba.cuda.tests

A summary of the unit test suite output is:

.. code-block::

   Ran 564 tests in 142.211s

   OK (skipped=11)

When running with the built-in Numba memory management, the output is:

.. code-block::

   Ran 564 tests in 133.396s

   OK (skipped=5)

i.e. the changes for using an external memory manager do not break the built-in
Numba memory management. There are an additional 6 skipped tests, from:


* ``TestDeallocation``\ : skipped as it specifically tests Numba's internal
  deallocation strategy.
* ``TestDeferCleanup``\ : skipped as it specifically tests Numba's implementation of
  deferred cleanup.
* ``TestCudaArrayInterface.test_ownership``\ : skipped as Numba does not own memory
  when an EMM Plugin is used, but ownership is assumed by this test case.

CuPy
~~~~

The test suite can be run with the CuPy plugin using:

.. code-block::

   NUMBA_CUDA_MEMORY_MANAGER=nbep7.cupy_mempool python -m numba.runtests numba.cuda.tests

This plugin implementation is presently more primitive than the RMM
implementation, and results in some errors with the unit test suite:

.. code-block::

   Ran 564 tests in 111.699s

   FAILED (errors=8, skipped=11)

The 8 errors are due to a lack of implementation of ``get_ipc_handle`` in the
CuPy EMM Plugin implementation. It is expected that this implementation will be
re-visited and completed so that CuPy can be used stably as an allocator for
Numba in the future.
