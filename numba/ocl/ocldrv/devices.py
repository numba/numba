"""
Runtime management layer on top of OpenCL driver
Allows Python style contexts, example:

    with ocl.platform('Intel') as plat:
        with plat.device('lake') as dev:
            with dev.current_context() as ctx:
                with ctx.current_queue() as que:

    ocl.platform('Intel').set_current()
    ocl.device('lake').set_current()
    ctx = ocl.current_context()
    que = ocl.current_queue()
"""
from __future__ import print_function, absolute_import, division
import functools
import threading
from numba import servicelib
from .driver import driver


class _ListWrapper(object):
    def _class_name(self):
        return self.__class__.__name__[1:-4].lower()

    def __getattr__(self, attr):
        if attr == "lst":
            self.lst = self._item_list
            return self.lst
        return super(_ListWrapper, self).__getattr__(attr)

    def __getitem__(self, num):
        return self.lst[num]

    def __str__(self):
        return ', '.join([str(d) for d in self.lst])

    def __iter__(self):
        return iter(self.lst)

    def __len__(self):
        return len(self.lst)

    @property
    def current(self):
        current_func = 'current_' + self._class_name()
        return getattr(_runtime, current_func)

class _StackWrapper(object):
    pass # @ ?

class _ManagerWrapper(object):
    def _class_name(self):
        return self.__class__.__name__[1:-7].lower()

    def __getattr__(self, attr):
        return getattr(self._item, attr)

    def __enter__(self):
        push_func = 'push_' + self._class_name()
        getattr(_runtime, push_func)(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        pop_func = 'pop_' + self._class_name()
        getattr(_runtime, pop_func)(self)

    def __str__(self):
        managed_class = "<Managed " + self._class_name()
        return managed_class + ": {self.name}>".format(self=self)

# ================================ OpenCL _Lists and _Managers

class _PlatformList(_ListWrapper):
    def __init__(self, driver):
        self._driver = driver

    @property
    def _item_list(self):
        return [_PlatformManager(plat) for plat in self._driver.platforms]

class _PlatformManager(_ManagerWrapper):
    def __init__(self, plat):
        self._platform = plat
        self.devices = _DeviceList(plat)

    @property
    def _item(self):
        return self._platform


class _DeviceList(_ListWrapper):
    def __init__(self, platform):
        self.platform = platform

    @property
    def _item_list(self):
        return [_DeviceManager(dev,self.platform) for dev in self.platform.all_devices]

class _DeviceManager(_ManagerWrapper):
    def __init__(self, device, platform):
        self._device = device
        self.platform = platform

    @property
    def _item(self):
        return self._device


class _ContextList(_ListWrapper):
    def __init__(self, device):
        self.device = device

    @property
    def _item_list(self):
        return [] # no context exists yet

class _ContextManager(_ManagerWrapper):
    def __init__(self, context,device):
        self._context = context
        self.device = device

    @property
    def _item(self):
        return self._context


class _TaskList(_ListWrapper):
    pass

class _TaskManager(_ManagerWrapper):
    pass

class _KernelList(_ListWrapper):
    pass

class _KernelManager(_ManagerWrapper):
    pass

class _QueueList(_ListWrapper):
    def __init__(self, context):
        self._context = context

    @property
    def _item_list(self):
        return [] # no context exists yet

class _QueueManager(_ManagerWrapper):
    def __init__(self, queue):
        self._queue = queue

    @property
    def _item(self):
        return self._queue

class _EventList(_ListWrapper):
    pass

class _EventManager(_ManagerWrapper):
    pass

# ================================ RUNTIME ===================================

class _Runtime(object):
    """OpenCL runtime management.
    It manages Platforms, Devices, Contexts, Queues, Tasks, Kernels, Events.
    Keeps at most one Context per Device, which is shared by the threads.
    Keeps at least:
        - one Queue per Context per thread
        - one Kernel per Task per thread
        - one Event per EnqueueCommand per thread ?
    """

    def __init__(self):
        # Avoid a race condition by which the driver releases before runtime
        self.driver = driver

        # List of OpenCL platforms
        self.lst = _PlatformList(driver)
        #self.devices = _DeviceList()
        #self.contexts = _ContextList()
        #self.tasks = _TaskList()
        #self.kernel = _KernelList()
        #self.queues = _QueueList()
        #self.events = _EventList()

        # A thread_local stack of items
        self.platform_stack = servicelib.TLStack()
        self.device_stack = servicelib.TLStack()
        self.context_stack = servicelib.TLStack()
        self.queue_stack = servicelib.TLStack()

        # Only the main thread can *actually* destroy
        self._mainthread = threading.current_thread()

        # Avoid mutation of runtime state in multithreaded programs
        self._lock = threading.RLock()

    def __del__(self):
        pass # delete all items before driver?

    @property
    def platforms(self):
        return self.lst

    @property
    def current_platform(self):
        if (self.platform_stack):
            return self.platform_stack.top

    def push_platform(self, plat):
        self.platform_stack.push(plat)

    def pop_platform(self, plat):
        assert (self.platform_stack.top == plat)
        #assert (len(self.platform_stack) > 1)
        self.platform_stack.pop()

    def get_or_select_platform(self):
        """Returns the current platform or select/push the first one"""
        if self.platform_stack:
            return self.current_platform
        else: # select/push first platform
            return self.select_platform(0)            

    def select_platform(self,id_or_name):
        """Selects a platform according to the given _id_ or _name_"""
        if isinstance(id_or_name, int):
            for i in range(len(self.platforms)):
                if i == id_or_name:
                    plat = self.platforms[i]
                    break
        elif isinstance(id_or_name, str):
            for plat in _runtime.platforms:
                if id_or_name in plat.name:
                    break
        else:
            assert False, "No platform was found!"
        plat = _PlatformManager(plat)
        self.push_platform(plat)
        return plat

    @property
    def devices(self):
       return self.current_platform.devices

    @property
    def current_device(self):
        if (self.device_stack):
            return self.device_stack.top

    def push_device(self, dev):
        self.device_stack.push(dev)

    def pop_device(self, dev):
        assert (self.device_stack.top == dev)
        #assert (len(self.device_stack) > 1)
        self.device_stack.pop()

    def get_or_select_device(self):
        """Returns the current device or select/push the first one"""
        if self.device_stack:
            return self.current_device
        else: # select/push first device
            return self.select_device(0)

    def select_device(self,id_or_name):
        """Selects a device according to the given _id_ or _name_"""
        if isinstance(id_or_name, int):
            for i in range(len(self.devices)):
                if i == id_or_name:
                    dev = self.devices[i]
                    break
        elif isinstance(id_or_name, str):
            for dev in _runtime.devices:
                if id_or_name in dev.name:
                    break
        else:
            assert False, "No device was found!"
        dev = _DeviceManager(dev,self.current_platform)
        self.push_device(dev)
        return dev

    @property
    def current_context(self):
        if (self.context_stack):
            return self.context_stack.top

    def push_context(self, ctx):
        self.context_stack.push(ctx)

    def pop_context(self, ctx):
        assert (self.context_stack.top == ctx)
        assert (len(self.context_stack) > 1)
        # Will not remove the last context in the stack
        if len(self.context_stack) > 1:
            self.context_stack.pop()

    def get_or_create_context(self):
        """Returns the current context or create/push a new one"""
        if self.context_stack:
            return self.current_context
        else: # push/create a context
            ctx = self._create_context()
            self.push_context(ctx) # @
            return ctx

    def _create_context(self):
        """Creates a new context with the current platform / device"""
        with self._lock:
            dev = self.current_device
            plat = dev.platform
            ctx = plat.create_context([dev])
            return _ContextManager(ctx,dev)

    @property
    def current_queue(self):
        if (self.queue_stack):
            return self.queue_stack.top

    def push_queue(self, ctx):
        self.queue_stack.push(ctx)

    def pop_queue(self, ctx):
        assert (self.queue_stack.top == ctx)
        assert (len(self.queue_stack) > 1)
        # Will not remove the last queue in the stack
        if len(self.queue_stack) > 1:
            self.queue_stack.pop()

    def get_or_create_queue(self):
        """Returns the current queue or create/push a new one"""
        if self.queue_stack:
            return self.current_queue
        else:  # push/create a queue
            que = self._create_queue()
            self.push_queue(que)
            return que

    def _create_queue(self):
        """Creates a new queue with the current context"""
        with self._lock:
            ctx = self.current_context
            dev = ctx.device
            que = ctx.create_command_queue(dev)
            return _QueueManager(que)

    def init(self):
        """Sets the first platform / device as default"""
        def_plat = self.platforms[0]
        def_dev = def_plat.devices[0]
        self.platform_stack.push(def_plat)
        self.device_stack.push(def_dev)

    def reset(self):
        """Clear the thread_local stacks. The main thread destroys the contexts
        """
        while self.platform_stack:
            ctx = self.platform_stack.pop()
        while self.device_stack:
            ctx = self.device_stack.pop()
        while self.context_stack:
            ctx = self.context_stack.pop()
        while self.queue_stack:
            ctx = self.queue_stack.pop()

        if threading.current_thread() == self._mainthread:
            self._destroy_all_contexts()

    def _destroy_all_contexts(self):
        # Reset all devices
        for plat in self.platforms:
            for dev in plat.devices:
                dev.reset()


# ================================ PUBLIC API ================================

def require_runtime(fn):
    @functools.wraps(fn)
    def _require_ocl_runtime(*args, **kws):
        assert(_runtime is not None)
        return fn(*args, **kws)
    return _require_ocl_runtime

def require_platform(fn):
    @functools.wraps(fn)
    def _require_ocl_platform(*args, **kws):
        with get_platform():
            return fn(*args, **kws)
    return _require_ocl_platform

def require_device(fn):
    @functools.wraps(fn)
    def _require_ocl_device(*args, **kws):
        with get_device():
            return fn(*args, **kws)
    return _require_ocl_device

def require_context(fn):
    @functools.wraps(fn)
    def _require_ocl_context(*args, **kws):
        with get_context():
            return fn(*args, **kws)
    return _require_ocl_context


@require_runtime
def get_platform():
    return _runtime.get_or_select_platform()

@require_platform
def get_device():
    return _runtime.get_or_select_device()

@require_device
def get_context():
    return _runtime.get_or_create_context()

@require_context
def get_queue():
    return _runtime.get_or_create_queue()


@require_runtime
def select_platform(id_or_name):
    return _runtime.select_platform(id_or_name)

@require_platform
def select_device(id_or_name):
    return _runtime.select_device(id_or_name)

#def create_context() ?

#def create_queue()

@require_runtime
def platforms():
    return _runtime.platforms

@require_platform
def devices():
    return _runtime.devices

# ================================ # init(), reset()

def init():
    """Initialize the OpenCL subsystem for the current thread"""
    _runtime.reset()
    _runtime.init()

    # @@ OpenCL 2.1
    for plat in _runtime.platforms:
        if "OpenCL 2.1" in plat.name:
            _runtime.push_platform(plat)
            _runtime.push_device(plat.devices[0])
            break
    else:
        print("NB: no OpenCL 2.1 platform was found!")

def reset():
    """Reset the OpenCL subsystem for the current thread.
    - In the main thread:
        This removes all OpenCL contexts.  Only use this at shutdown or for
        cleaning up between tests.
    - In non-main threads:
        This clear the OpenCL context stack only.
    """
    _runtime.reset()

import atexit
atexit.register(reset)

# ================================ # Exposed variables

_runtime = _Runtime()
init() # initializing OpenCL 2.1

gpus = _runtime.devices
