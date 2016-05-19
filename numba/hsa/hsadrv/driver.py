"""
HSA driver bridge implementation
"""

from __future__ import absolute_import, print_function, division

import sys
import atexit
import os
import ctypes
import struct
import weakref
from collections import Sequence
from numba.utils import total_ordering
from numba import utils
from numba import config
from .error import HsaSupportError, HsaDriverError, HsaApiError
from . import enums, drvapi


class HsaKernelTimedOut(HsaDriverError):
    pass


def _device_type_to_string(device):
    try:
        return ['CPU', 'GPU', 'DSP'][device]
    except IndexError:
        return 'Unknown'


DEFAULT_HSA_DRIVER = '/opt/hsa/lib/libhsa-runtime64.so'


def _find_driver():
    envpath = os.environ.get('NUMBA_HSA_DRIVER', DEFAULT_HSA_DRIVER)
    if envpath == '0':
        # Force fail
        _raise_driver_not_found()

    # Determine DLL type
    if (struct.calcsize('P') != 8
        or sys.platform == 'win32'
        or sys.platform == 'darwin'):
        _raise_platform_not_supported()
    else:
        # Assume to be *nix like and 64 bit
        dlloader = ctypes.CDLL
        dldir = ['/usr/lib', '/usr/lib64']
        dlname = 'libhsa-runtime64.so'

    if envpath is not None:
        try:
            envpath = os.path.abspath(envpath)
        except ValueError:
            raise ValueError("NUMBA_HSA_DRIVER %s is not a valid path" %
                             envpath)
        if not os.path.isfile(envpath):
            raise ValueError("NUMBA_HSA_DRIVER %s is not a valid file "
                             "path.  Note it must be a filepath of the .so/"
                             ".dll/.dylib or the driver" % envpath)
        candidates = [envpath]
    else:
        # First search for the name in the default library path.
        # If that is not found, try the specific path.
        candidates = [dlname] + [os.path.join(x, dlname) for x in dldir]

    # Load the driver; Collect driver error information
    path_not_exist = []
    driver_load_error = []

    for path in candidates:
        try:
            dll = dlloader(path)
        except OSError as e:
            # Problem opening the DLL
            path_not_exist.append(not os.path.isfile(path))
            driver_load_error.append(e)
        else:
            return dll

    # Problem loading driver
    if all(path_not_exist):
        _raise_driver_not_found()
    else:
        errmsg = '\n'.join(str(e) for e in driver_load_error)
        _raise_driver_error(errmsg)


PLATFORM_NOT_SUPPORTED_ERROR = """
HSA is not currently supported on this platform ({0}).
"""


def _raise_platform_not_supported():
    raise HsaSupportError(PLATFORM_NOT_SUPPORTED_ERROR.format(sys.platform))


DRIVER_NOT_FOUND_MSG = """
The HSA runtime library cannot be found.

If you are sure that the HSA is installed, try setting environment
variable NUMBA_HSA_DRIVER with the file path of the HSA runtime shared
library.
"""


def _raise_driver_not_found():
    raise HsaSupportError(DRIVER_NOT_FOUND_MSG)


DRIVER_LOAD_ERROR_MSG = """
A HSA runtime library was found, but failed to load with error:
%s
"""


def _raise_driver_error(e):
    raise HsaSupportError(DRIVER_LOAD_ERROR_MSG % e)


MISSING_FUNCTION_ERRMSG = """driver missing function: %s.
"""


class Recycler(object):
    def __init__(self):
        self._garbage = []
        self.enabled = True

    def free(self, obj):
        self._garbage.append(obj)
        self.service()

    def _cleanup(self):
        for obj in self._garbage:
            obj._finalizer(obj)
        del self._garbage[:]

    def service(self):
        if self.enabled:
            if len(self._garbage) > 10:
                self._cleanup()

    def drain(self):
        self._cleanup()
        self.enabled = False


# The Driver ###########################################################


class Driver(object):
    """
    Driver API functions are lazily bound.
    """
    _singleton = None
    _agent_map = None
    _api_prototypes = drvapi.API_PROTOTYPES  # avoid premature GC at exit

    _hsa_properties = {
        'version_major': (enums.HSA_SYSTEM_INFO_VERSION_MAJOR, ctypes.c_uint16),
        'version_minor': (enums.HSA_SYSTEM_INFO_VERSION_MINOR, ctypes.c_uint16),
        'timestamp': (enums.HSA_SYSTEM_INFO_TIMESTAMP, ctypes.c_uint64),
        'timestamp_frequency': (enums.HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, ctypes.c_uint16),
        'signal_max_wait': (enums.HSA_SYSTEM_INFO_SIGNAL_MAX_WAIT, ctypes.c_uint64),
    }

    def __new__(cls):
        obj = cls._singleton
        if obj is not None:
            return obj
        else:
            obj = object.__new__(cls)
            cls._singleton = obj
        return obj

    def __init__(self):
        try:
            if config.DISABLE_HSA:
                raise HsaSupportError("HSA disabled by user")
            self.lib = _find_driver()
            self.is_initialized = False
            self.initialization_error = None
        except HsaSupportError as e:
            self.is_initialized = True
            self.initialization_error = e

        self._agent_map = None
        self._programs = {}
        self._recycler = Recycler()

    def _initialize_api(self):
        if self.is_initialized:
            return

        self.is_initialized = True
        try:
            self.hsa_init()
        except HsaApiError as e:
            self.initialization_error = e
            raise HsaDriverError("Error at driver init: \n%s:" % e)
        else:
            @atexit.register
            def shutdown():
                for agent in self.agents:
                    agent.release()
                self._recycler.drain()

    def _initialize_agents(self):
        if self._agent_map is not None:
            return

        self._initialize_api()

        agent_ids = []

        def on_agent(agent_id, ctxt):
            agent_ids.append(agent_id)
            return enums.HSA_STATUS_SUCCESS

        callback = drvapi.HSA_ITER_AGENT_CALLBACK_FUNC(on_agent)
        self.hsa_iterate_agents(callback, None)

        agent_map = dict((agent_id, Agent(agent_id)) for agent_id in agent_ids)
        self._agent_map = agent_map

    @property
    def is_available(self):
        self._initialize_api()
        return self.initialization_error is None

    @property
    def agents(self):
        self._initialize_agents()
        return self._agent_map.values()

    def create_program(self, model=enums.HSA_MACHINE_MODEL_LARGE,
                       profile=enums.HSA_PROFILE_FULL,
                       rounding_mode=enums.HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                       options=None):
        program = drvapi.hsa_ext_program_t()
        assert options is None
        self.hsa_ext_program_create(model, profile, rounding_mode,
                                    options, ctypes.byref(program))
        return Program(program)

    def create_signal(self, initial_value, consumers=None):
        if consumers is not None:
            consumers_len = len(consumers)
            consumers_type = drvapi.hsa_agent_t * consumers_len
            consumers = consumers_type(*[c._id for c in consumers])
        else:
            consumers_len = 0

        result = drvapi.hsa_signal_t()
        self.hsa_signal_create(initial_value, consumers_len, consumers,
                               ctypes.byref(result))
        return Signal(result.value)

    # def load_code_unit(self, code_binary, agents=None):
    # # not sure of the purpose of caller...
    #     caller = drvapi.hsa_runtime_caller_t()
    #     caller.caller = 0
    #
    #     if agents is not None:
    #         agent_count = len(agents)
    #         agents = (drvapi.hsa_agent_t * agent_count)(*agents)
    #     else:
    #         agent_count = 0
    #
    #     # callback not yet supported, always use NULL
    #     cb = ctypes.cast(None, drvapi.hsa_ext_symbol_value_callback_t)
    #
    #     result = drvapi.hsa_code_unit_t()
    #     self.hsa_ext_code_unit_load(caller, agents, agent_count, code_binary,
    #                                 len(code_binary), options, cb,
    #                                 ctypes.byref(result))
    #
    #     return CodeUnit(result)

    def __getattr__(self, fname):
        # Initialize driver
        self._initialize_api()

        # First try if it is an hsa property
        try:
            enum, typ = self._hsa_properties[fname]
            result = typ()
            self.hsa_system_get_info(enum, ctypes.byref(result))
            return result.value
        except KeyError:
            pass

        # if not a property... try if it is an api call
        try:
            proto = self._api_prototypes[fname]
        except KeyError:
            raise AttributeError(fname)

        if self.initialization_error is not None:
            raise HsaSupportError("Error at driver init: \n%s:" %
                                  self.initialization_error)

        # Find function in driver library
        libfn = self._find_api(fname)

        for key, val in proto.items():
            setattr(libfn, key, val)

        def driver_wrapper(fn):
            def wrapped(*args, **kwargs):
                return fn(*args, **kwargs)

            return wrapped

        retval = driver_wrapper(libfn)
        setattr(self, fname, retval)
        return retval

    def _find_api(self, fname):
        # Try regular
        try:
            return getattr(self.lib, fname)
        except AttributeError:
            pass

        # Not found.
        # Delay missing function error to use
        def absent_function(*args, **kws):
            raise HsaDriverError(MISSING_FUNCTION_ERRMSG % fname)

        setattr(self, fname, absent_function)
        return absent_function

    @property
    def components(self):
        """Returns a ordered list of components

        The first device should be picked first
        """
        return list(filter(lambda a: a.is_component, reversed(sorted(
            self.agents))))


hsa = Driver()


class HsaWrapper(object):
    def __getattr__(self, fname):
        try:
            enum, typ = self._hsa_properties[fname]
        except KeyError:
            raise AttributeError(
                "%r object has no attribute %r" % (self.__class__, fname))

        func = getattr(hsa, self._hsa_info_function)
        result = typ()
        is_array_type = hasattr(typ, '_length_')
        # if the result is not ctypes array, get a reference)
        result_buff = result if is_array_type else ctypes.byref(result)
        func(self._id, enum, result_buff)

        if not is_array_type or typ._type_ == ctypes.c_char:
            return result.value
        else:
            return list(result)

    def __dir__(self):
        return sorted(set(dir(type(self)) +
                          self.__dict__.keys() +
                          self._hsa_properties.keys()))


@total_ordering
class Agent(HsaWrapper):
    """Abstracts a HSA compute agent.

    This will wrap and provide an OO interface for hsa_agent_t C-API elements
    """

    # Note this will be handled in a rather unconventional way. When agents get
    # initialized by the driver, a set of instances for all the available agents
    # will be created. After that creation, the __new__ and __init__ methods will
    # be replaced, and the constructor will act as a mapping from an agent_id to
    # the equivalent Agent object. Any attempt to create an Agent with a non
    # existing agent_id will result in an error.
    #
    # the logic for this resides in Driver._initialize_agents

    _hsa_info_function = 'hsa_agent_get_info'
    _hsa_properties = {
        'name': (enums.HSA_AGENT_INFO_NAME, ctypes.c_char * 64),
        'vendor_name': (enums.HSA_AGENT_INFO_VENDOR_NAME, ctypes.c_char * 64),
        'feature': (enums.HSA_AGENT_INFO_FEATURE, drvapi.hsa_agent_feature_t),
        'wavefront_size': (
            enums.HSA_AGENT_INFO_WAVEFRONT_SIZE, ctypes.c_uint32),
        'workgroup_max_dim': (
            enums.HSA_AGENT_INFO_WORKGROUP_MAX_DIM, ctypes.c_uint16 * 3),
        'grid_max_dim': (enums.HSA_AGENT_INFO_GRID_MAX_DIM, drvapi.hsa_dim3_t),
        'grid_max_size': (enums.HSA_AGENT_INFO_GRID_MAX_SIZE, ctypes.c_uint32),
        'fbarrier_max_size': (
            enums.HSA_AGENT_INFO_FBARRIER_MAX_SIZE, ctypes.c_uint32),
        'queues_max': (enums.HSA_AGENT_INFO_QUEUES_MAX, ctypes.c_uint32),
        'queue_max_size': (
            enums.HSA_AGENT_INFO_QUEUE_MAX_SIZE, ctypes.c_uint32),
        'queue_type': (
            enums.HSA_AGENT_INFO_QUEUE_TYPE, drvapi.hsa_queue_type_t),
        'node': (enums.HSA_AGENT_INFO_NODE, ctypes.c_uint32),
        '_device': (enums.HSA_AGENT_INFO_DEVICE, drvapi.hsa_device_type_t),
        'cache_size': (enums.HSA_AGENT_INFO_CACHE_SIZE, ctypes.c_uint32 * 4),
        'isa': (enums.HSA_AGENT_INFO_ISA, drvapi.hsa_isa_t),
    }

    def __init__(self, agent_id):
        # This init will only happen when initializing the agents. After
        # the agent initialization the instances of this class are considered
        # initialized and locked, so this method will be removed.
        self._id = agent_id
        self._recycler = hsa._recycler
        self._queues = set()
        self._initialize_regions()

    @property
    def device(self):
        return _device_type_to_string(self._device)

    @property
    def is_component(self):
        return (self.feature & enums.HSA_AGENT_FEATURE_KERNEL_DISPATCH) != 0

    @property
    def regions(self):
        return self._regions

    def _initialize_regions(self):
        region_ids = []

        def on_region(region_id, ctxt):
            region_ids.append(region_id)
            return enums.HSA_STATUS_SUCCESS

        callback = drvapi.HSA_AGENT_ITERATE_REGIONS_CALLBACK_FUNC(on_region)
        hsa.hsa_agent_iterate_regions(self._id, callback, None)
        self._regions = _RegionList([MemRegion.instance_for(self, region_id)
                                     for region_id in region_ids])

    def _create_queue(self, size, callback=None, data=None,
                      private_segment_size=None, group_segment_size=None,
                      queue_type=None):
        assert queue_type is not None
        assert size <= self.queue_max_size
        cb_typ = drvapi.HSA_QUEUE_CALLBACK_FUNC
        cb = ctypes.cast(None, cb_typ) if callback is None else cb_typ(callback)
        result = ctypes.POINTER(drvapi.hsa_queue_t)()
        private_segment_size = (ctypes.c_uint32(-1)
                                if private_segment_size is None
                                else private_segment_size)
        group_segment_size = (ctypes.c_uint32(-1)
                              if group_segment_size is None
                              else group_segment_size)
        hsa.hsa_queue_create(self._id, size, queue_type, cb, data,
                             private_segment_size, group_segment_size,
                             ctypes.byref(result))

        q = Queue(self, result)
        self._queues.add(q)
        return weakref.proxy(q)

    def create_queue_single(self, *args, **kwargs):
        kwargs['queue_type'] = enums.HSA_QUEUE_TYPE_SINGLE
        return self._create_queue(*args, **kwargs)

    def create_queue_multi(self, *args, **kwargs):
        kwargs['queue_type'] = enums.HSA_QUEUE_TYPE_MULTI
        return self._create_queue(*args, **kwargs)

    def release(self):
        """
        Release all resources

        Called at system teardown
        """
        for q in list(self._queues):
            q.release()

    def release_queue(self, queue):
        self._queues.remove(queue)
        self._recycler.free(queue)

    def __repr__(self):
        return "<HSA agent ({0}): {1} {2} '{3}'{4}>".format(self._id,
                                                            self.device,
                                                            self.vendor_name,
                                                            self.name,
                                                            " (component)" if self.is_component else "")

    def _rank(self):
        return (self.is_component, self.grid_max_size, self._device)

    def __lt__(self, other):
        if isinstance(self, Agent):
            return self._rank() < other._rank()
        else:
            return NotImplemented

    def __eq__(self, other):
        if isinstance(self, Agent):
            return self._rank() == other._rank()
        else:
            return NotImplemented

    def __hash__(self):
        return hash(self._rank())

    def create_context(self):
        return Context(self)


class _RegionList(Sequence):
    __slots__ = '_all', 'globals', 'readonlys', 'privates', 'groups'

    def __init__(self, lst):
        self._all = tuple(lst)
        self.globals = tuple(x for x in lst if x.kind == 'global')
        self.readonlys = tuple(x for x in lst if x.kind == 'readonly')
        self.privates = tuple(x for x in lst if x.kind == 'private')
        self.groups = tuple(x for x in lst if x.kind == 'group')

    def __len__(self):
        return len(self._all)

    def __contains__(self, item):
        return item in self._all

    def __reversed__(self):
        return reversed(self._all)

    def __getitem__(self, idx):
        return self._all[idx]


class MemRegion(HsaWrapper):
    """Abstracts a HSA memory region.

    This will wrap and provide an OO interface for hsa_region_t C-API elements
    """
    _hsa_info_function = 'hsa_region_get_info'
    _hsa_properties = {
        'segment': (
            enums.HSA_REGION_INFO_SEGMENT,
            drvapi.hsa_region_segment_t
        ),
        '_flags': (
            enums.HSA_REGION_INFO_GLOBAL_FLAGS,
            drvapi.hsa_region_flag_t
        ),
        'size': (enums.HSA_REGION_INFO_SIZE,
                 ctypes.c_size_t),
        'alloc_max_size': (enums.HSA_REGION_INFO_ALLOC_MAX_SIZE,
                           ctypes.c_size_t),
        'alloc_alignment': (enums.HSA_REGION_INFO_RUNTIME_ALLOC_ALIGNMENT,
                            ctypes.c_size_t),
        'alloc_granule': (enums.HSA_REGION_INFO_RUNTIME_ALLOC_GRANULE,
                          ctypes.c_size_t),
        'alloc_allowed': (enums.HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED,
                          ctypes.c_bool),
    }

    _segment_name_map = {
        enums.HSA_REGION_SEGMENT_GLOBAL: 'global',
        enums.HSA_REGION_SEGMENT_READONLY: 'readonly',
        enums.HSA_REGION_SEGMENT_PRIVATE: 'private',
        enums.HSA_REGION_SEGMENT_GROUP: 'group',
    }

    def __init__(self, agent, region_id):
        """Do not instantiate MemRegion objects directly, use the factory class
        method 'instance_for' to ensure MemRegion identity"""
        self._id = region_id
        self._owner_agent = agent
        self._kind = self._segment_name_map[self.segment]

    @property
    def kind(self):
        return self._kind

    @property
    def agent(self):
        return self._owner_agent

    @property
    def supports_kernargs(self):
        if self.kind == 'global':
            return self._flags & enums.HSA_REGION_GLOBAL_FLAG_KERNARG
        else:
            return False

    def allocate(self, ctypes_type):
        assert self.alloc_allowed
        alloc_size = ctypes.sizeof(ctypes_type)
        assert alloc_size <= self.alloc_max_size
        buff = ctypes.c_void_p()
        hsa.hsa_memory_allocate(self._id, alloc_size,
                                ctypes.byref(buff))
        return ctypes_type.from_address(buff.value)

    def free(self, ptr):
        hsa.hsa_memory_free(ptr)

    _instance_dict = {}

    @classmethod
    def instance_for(cls, owner, _id):
        try:
            return cls._instance_dict[_id]
        except KeyError:
            new_instance = cls(owner, _id)
            cls._instance_dict[_id] = new_instance
            return new_instance


class Queue(object):
    def __init__(self, agent, queue_ptr):
        """The id in a queue is a pointer to the queue object returned by hsa_queue_create.
        The Queue object has ownership on that queue object"""
        self._agent = weakref.proxy(agent)
        self._id = queue_ptr
        self._as_parameter_ = self._id
        self._finalizer = hsa.hsa_queue_destroy

    def release(self):
        self._agent.release_queue(self)

    def __getattr__(self, fname):
        return getattr(self._id.contents, fname)

    def dispatch(self, symbol, kernargs,
                 workgroup_size=None,
                 grid_size=None,
                 signal=None):
        dims = len(workgroup_size)
        assert dims == len(grid_size)
        assert 0 < dims <= 3
        assert grid_size >= workgroup_size
        if workgroup_size > tuple(self._agent.workgroup_max_dim)[:dims]:
            msg = "workgroupsize is too big {0} > {1}"
            raise HsaDriverError(msg.format(workgroup_size,
                                tuple(self._agent.workgroup_max_dim)[:dims]))
        s = signal if signal is not None else hsa.create_signal(1)

        # Note: following vector_copy.c

        # Obtain the current queue write index
        index = hsa.hsa_queue_load_write_index_relaxed(self._id)

        # Write AQL packet at the calculated queue index address
        queue_struct = self._id.contents
        queue_mask = queue_struct.size - 1

        dispatch_packet_t = drvapi.hsa_kernel_dispatch_packet_t
        packet_array_t = (dispatch_packet_t * queue_struct.size)

        queue_offset = index & queue_mask
        queue = packet_array_t.from_address(queue_struct.base_address)

        packet = queue[queue_offset]

        # Populate packet
        packet.setup |= dims << enums.HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS

        packet.workgroup_size_x = workgroup_size[0]
        packet.workgroup_size_y = workgroup_size[1] if dims > 1 else 1
        packet.workgroup_size_z = workgroup_size[2] if dims > 2 else 1

        packet.grid_size_x = grid_size[0]
        packet.grid_size_y = grid_size[1] if dims > 1 else 1
        packet.grid_size_z = grid_size[2] if dims > 2 else 1

        packet.completion_signal = s._id

        packet.kernel_object = symbol.kernel_object

        packet.kernarg_address = (0 if kernargs is None
                                  else ctypes.addressof(kernargs))

        packet.private_segment_size = symbol.private_segment_size
        packet.group_segment_size = symbol.group_segment_size

        header = 0
        header |= enums.HSA_FENCE_SCOPE_SYSTEM << enums.HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE
        header |= enums.HSA_FENCE_SCOPE_SYSTEM << enums.HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE
        header |= enums.HSA_PACKET_TYPE_KERNEL_DISPATCH << enums.HSA_PACKET_HEADER_TYPE

        # Original example calls for an atomic store.
        # Since we are on x86, store of aligned 16 bit is atomic.
        # The C code is
        # __atomic_store_n((uint16_t*)(&dispatch_packet->header), header, __ATOMIC_RELEASE);
        packet.header = header

        # Increment write index
        hsa.hsa_queue_store_write_index_relaxed(self._id, index + 1)
        # Ring the doorbell
        hsa.hsa_signal_store_relaxed(self._id.contents.doorbell_signal, index)

        # Wait on the dispatch completion signal

        # synchronous if no signal was provided
        if signal is None:
            timeout = 10
            if not s.wait_until_ne_one(timeout=timeout):
                msg = "Kernel timed out after {timeout} second"
                raise HsaKernelTimedOut(msg.format(timeout=timeout))

    def __dir__(self):
        return sorted(set(dir(self._id.contents) +
                          self.__dict__.keys()))

    def owned(self):
        return ManagedQueueProxy(self)


class ManagedQueueProxy(object):
    def __init__(self, queue):
        self._queue = weakref.ref(queue)

    def __getattr__(self, item):
        return getattr(self._queue(), item)


class Signal(object):
    """The id for the signal is going to be the hsa_signal_t returned by create_signal.
    Lifetime of the underlying signal will be tied with this object".
    Note that it is likely signals will have lifetime issues."""

    def __init__(self, signal_id):
        self._id = signal_id
        self._as_parameter_ = self._id
        utils.finalize(self, hsa.hsa_signal_destroy, self._id)

    def load_relaxed(self):
        return hsa.hsa_signal_load_relaxed(self._id)

    def load_acquired(self):
        return hsa.hsa_signal_load_acquired(self._id)

    def wait_until_ne_one(self, timeout=None):
        """
        Returns a boolean to indicate whether the wait has timeout
        """
        one = 1
        mhz = 10 ** 6
        if timeout is None:
            # Infinite
            expire = -1   # UINT_MAX
        else:
            # timeout as seconds
            expire = timeout * hsa.timestamp_frequency * mhz
        hsa.hsa_signal_wait_acquire(self._id, enums.HSA_SIGNAL_CONDITION_NE,
                                    one, expire,
                                    enums.HSA_WAIT_STATE_BLOCKED)
        return self.load_relaxed() != one


class BrigModule(object):
    def __init__(self, brig_buffer):
        """
        Take a byte buffer of a Brig module
        """
        buf = ctypes.create_string_buffer(brig_buffer)
        self._buffer = buf
        self._id = ctypes.cast(ctypes.addressof(buf),
                               drvapi.hsa_ext_module_t)

    @classmethod
    def from_file(cls, file_name):
        with open(file_name, 'rb') as fin:
            buf = fin.read()

        return BrigModule(buf)

    def __len__(self):
        return len(self._buffer)

    def __repr__(self):
        return "<BrigModule id={0} size={1}bytes>".format(hex(id(self)),
                                                          len(self))


class Program(object):
    def __init__(self, model=enums.HSA_MACHINE_MODEL_LARGE,
                 profile=enums.HSA_PROFILE_FULL,
                 rounding_mode=enums.HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                 options=None):
        self._id = drvapi.hsa_ext_program_t()
        assert options is None
        hsa.hsa_ext_program_create(model, profile, rounding_mode,
                                   options, ctypes.byref(self._id))
        self._as_parameter_ = self._id
        utils.finalize(self, hsa.hsa_ext_program_destroy, self._id)

    def add_module(self, module):
        hsa.hsa_ext_program_add_module(self._id, module._id)

    def finalize(self, isa, callconv=0, options=None):
        """
        The program object is safe to be deleted after ``finalize``.
        """
        code_object = drvapi.hsa_code_object_t()
        control_directives = drvapi.hsa_ext_control_directives_t()
        ctypes.memset(ctypes.byref(control_directives), 0,
                      ctypes.sizeof(control_directives))
        hsa.hsa_ext_program_finalize(self._id,
                                     isa,
                                     callconv,
                                     control_directives,
                                     options,
                                     enums.HSA_CODE_OBJECT_TYPE_PROGRAM,
                                     ctypes.byref(code_object))
        return CodeObject(code_object)


class CodeObject(object):
    def __init__(self, code_object):
        self._id = code_object
        self._as_parameter_ = self._id
        utils.finalize(self, hsa.hsa_code_object_destroy, self._id)


class Executable(object):
    def __init__(self):
        ex = drvapi.hsa_executable_t()
        hsa.hsa_executable_create(enums.HSA_PROFILE_FULL,
                                  enums.HSA_EXECUTABLE_STATE_UNFROZEN,
                                  None,
                                  ctypes.byref(ex))
        self._id = ex
        self._as_parameter_ = self._id
        utils.finalize(self, hsa.hsa_executable_destroy, self._id)

    def load(self, agent, code_object):
        hsa.hsa_executable_load_code_object(self._id, agent._id,
                                            code_object._id, None)

    def freeze(self):
        """Freeze executable before we can query for symbol"""
        hsa.hsa_executable_freeze(self._id, None)

    def get_symbol(self, agent, name):
        symbol = drvapi.hsa_executable_symbol_t()
        hsa.hsa_executable_get_symbol(self._id, None,
                                      ctypes.create_string_buffer(
                                          name.encode('ascii')),
                                      agent._id, 0,
                                      ctypes.byref(symbol))
        return Symbol(symbol)


class Symbol(HsaWrapper):
    _hsa_info_function = 'hsa_executable_symbol_get_info'
    _hsa_properties = {
        'kernel_object': (
            enums.HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
            ctypes.c_uint64,
        ),
        'kernarg_segment_size': (
            enums.HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
            ctypes.c_uint32,
        ),
        'group_segment_size': (
            enums.HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
            ctypes.c_uint32,
        ),
        'private_segment_size': (
            enums.HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
            ctypes.c_uint32,
        ),
    }

    def __init__(self, symbol_id):
        self._id = symbol_id


class Context(object):
    """
    A context is associated with a component
    """

    def __init__(self, agent):
        self._agent = weakref.proxy(agent)
        qs = agent.queue_max_size
        defq = self._agent.create_queue_multi(qs, callback=self._callback)
        self._defaultqueue = defq.owned()

    def _callback(self, status, queue):
        drvapi._check_error(status, queue)
        sys.exit(1)

    @property
    def default_queue(self):
        return self._defaultqueue

    @property
    def agent(self):
        return self._agent
