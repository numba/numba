"""
HSA driver bridge implementation
"""

from __future__ import absolute_import, print_function, division

import sys
import os
import ctypes
import struct

from ... import config # base numba
from .error import HsaSupportError, HsaDriverError, HsaApiError, HsaWarning
from . import enums, drvapi, elf_utils

def _device_type_to_string(device):
    try:
        return ['CPU', 'GPU', 'DSP'][device]
    except IndexError:
        return 'Unknown'


def _find_driver():
    envpath = os.environ.get('NUMBA_HSA_DRIVER', None)
    if envpath == '0':
        # Force fail
        _raise_driver_not_found()

    # Determine DLL type
    if struct.calcsize('P') != 8 or sys.platform == 'win32' or sys.platform == 'darwin':
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
HSA is not currently ussported in this platform ({0}).
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

# The Driver ###########################################################
class Driver(object):
    """
    Driver API functions are lazily bound.
    """
    _singleton = None
    _agent_map = None

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


    def __del__(self):
        if self.is_initialized and self.initialization_error is None:
            self.hsa_shut_down()


    def _initialize_api(self):
        if self.is_initialized:
            return

        self.is_initialized = True
        try:
            self.hsa_init()
        except HsaApiError as e:
            self.initialization_error = e
            raise HsaDriverError("Error at driver init: \n%s:" % e)


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

        # del(Agent.__new__)
        agent_map = { agent_id: Agent(agent_id) for agent_id in agent_ids }
        del(Agent.__init__)
        @classmethod
        def _get_agent(_, _2, agent_id):
            try:
                return self._agent_map[agent_id]
            except KeyError:
                raise HsaDriverError("No known agent with id {0}".format(agent_id))

        self._agent_map = agent_map
        Agent.__new__ = _get_agent


    @property
    def is_available(self):
        self._initialize_api()
        return self.initialization_error is None


    @property
    def agents(self):
        self._initialize_agents()
        return self._agent_map.values()

    def create_program(self, device_list,
                       model=enums.HSA_EXT_BRIG_MACHINE_LARGE,
                       profile=enums.HSA_EXT_BRIG_PROFILE_FULL):
        device_list_len = len(device_list)
        device_list_type = drvapi.hsa_agent_t * device_list_len
        devices = device_list_type(*[d._id for d in device_list])
        program = drvapi.hsa_ext_program_handle_t()
        self.hsa_ext_program_create(devices, device_list_len, model, profile,
                                    ctypes.byref(program))
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

    def load_code_unit(self, code_binary, agents=None):
        # not sure of the purpose of caller...
        caller = drvapi.hsa_runtime_caller_t()
        caller.caller = 0

        if agents is not None:
            agent_count = len(agents)
            agents = (drvapi.hsa_agent_t * agent_count)(*agents)
        else:
            agent_count = 0

        # callback not yet supported, always use NULL
        cb = ctypes.cast(None, drvapi.hsa_ext_symbol_value_callback_t)

        result = drvapi.hsa_code_unit_t()
        self.hsa_ext_code_unit_load(caller, agents, agent_count, code_binary,
                                    len(code_binary), options, cb,
                                    ctypes.byref(result))

        return CodeUnit(result)


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
            proto = drvapi.API_PROTOTYPES[fname]
        except KeyError:
            raise AttributeError(fname)

        if self.initialization_error is not None:
            raise HsaSupportError("Error at driver init: \n%s:" %
                                  self.initialization_error)

        # Find function in driver library
        libfn = self._find_api(fname)

        for key, val in proto.items():
            setattr(libfn, key, val)

        setattr(self, fname, libfn)
        return libfn


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


hsa = Driver()

class HsaWrapper(object):
    def __getattr__(self, fname):
        try:
            enum, typ = self._hsa_properties[fname]
        except KeyError:
            raise AttributeError("%r object has no attribute %r" % (self.__class__, fname))

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
        'wavefront_size': (enums.HSA_AGENT_INFO_WAVEFRONT_SIZE, ctypes.c_uint32),
        'workgroup_max_dim': (enums.HSA_AGENT_INFO_WORKGROUP_MAX_DIM, ctypes.c_uint16 * 3),
        'grid_max_dim': (enums.HSA_AGENT_INFO_GRID_MAX_DIM, drvapi.hsa_dim3_t),
        'grid_max_size': (enums.HSA_AGENT_INFO_GRID_MAX_SIZE, ctypes.c_uint32),
        'fbarrier_max_size': (enums.HSA_AGENT_INFO_FBARRIER_MAX_SIZE, ctypes.c_uint32),
        'queues_max': (enums.HSA_AGENT_INFO_QUEUES_MAX, ctypes.c_uint32),
        'queue_max_size': (enums.HSA_AGENT_INFO_QUEUE_MAX_SIZE, ctypes.c_uint32),
        'queue_type': (enums.HSA_AGENT_INFO_QUEUE_TYPE, drvapi.hsa_queue_type_t),
        'node': (enums.HSA_AGENT_INFO_NODE, ctypes.c_uint32),
        '_device': (enums.HSA_AGENT_INFO_DEVICE, drvapi.hsa_device_type_t),
        'cache_size': (enums.HSA_AGENT_INFO_CACHE_SIZE, ctypes.c_uint32 * 4),
        'image1d_max_dim': (enums.HSA_EXT_AGENT_INFO_IMAGE1D_MAX_DIM, drvapi.hsa_dim3_t),
        'image2d_max_dim': (enums.HSA_EXT_AGENT_INFO_IMAGE2D_MAX_DIM, drvapi.hsa_dim3_t),
        'image3d_max_dim': (enums.HSA_EXT_AGENT_INFO_IMAGE3D_MAX_DIM, drvapi.hsa_dim3_t),
        'image_array_max_size': (enums.HSA_EXT_AGENT_INFO_IMAGE_ARRAY_MAX_SIZE, ctypes.c_uint32),
        'image_rd_max': (enums.HSA_EXT_AGENT_INFO_IMAGE_RD_MAX, ctypes.c_uint32),
        'image_rdwr_max': (enums.HSA_EXT_AGENT_INFO_IMAGE_RDWR_MAX, ctypes.c_uint32),
        'sampler_max': (enums.HSA_EXT_AGENT_INFO_SAMPLER_MAX, ctypes.c_uint32),
    }


    # def __new__(cls, agent_id):
    #     # This is here to raise errors when trying to create agents
    #     # before initialization. When agents are initialized, __new__ will
    #     # be replaced with a version that returns the appropriate instance
    #     # for existing agent_ids
    #     raise HsaDriverError("No known agent with id {0}".format(agent_id))


    def __init__(self, agent_id):
        # This init will only happen when initializing the agents. After
        # the agent initialization the instances of this class are considered
        # initialized and locked, so this method will be removed.
        self._id = agent_id
        self._initialize_regions()

    @property
    def device(self):
        return _device_type_to_string(self._device)


    @property
    def is_component(self):
        return (self.feature & enums.HSA_AGENT_FEATURE_DISPATCH) != 0

    @property
    def regions(self):
        return self._regions

    def _initialize_regions(self):
        region_ids = []

        def on_region(agent_id, ctxt):
            region_ids.append(agent_id)
            return enums.HSA_STATUS_SUCCESS

        callback = drvapi.HSA_AGENT_ITERATE_REGIONS_CALLBACK_FUNC(on_region)
        hsa.hsa_agent_iterate_regions(self._id, callback, None)
        self._regions = [MemRegion.instance_for(region_id)
                         for region_id in region_ids]

    def create_queue_single(self, size, callback=None, service_queue=None):
        cb_typ = drvapi.HSA_QUEUE_CALLBACK_FUNC
        cb = ctypes.cast(None, cb_typ) if callback is None else cb_typ(callback)
        sq = None if service_queue is None else service_queue._id
        result = ctypes.POINTER(drvapi.hsa_queue_t)()
        hsa.hsa_queue_create(self._id, size, enums.HSA_QUEUE_TYPE_SINGLE,
                             cb, sq, ctypes.byref(result))
        return Queue(result)


    def create_queue_multi(self, size, callback=None, service_queue=None):
        cb_typ = drvapi.HSA_QUEUE_CALLBACK_FUNC
        cb = ctypes.cast(None, cb_typ) if callback is None else cb_typ(callback)
        sq = None if service_queue is None else service_queue._id
        result = ctypes.POINTER(drvapi.hsa_queue_t)()
        hsa.hsa_queue_create(self._id, size, enums.HSA_QUEUE_TYPE_MULTI,
                             cb, sq, ctypes.byref(result))
        return Queue(result)


    def __repr__(self):
        return "<HSA agent ({0}): {1} {2} '{3}'{4}>".format(self._id, self.device,
                                                            self.vendor_name, self.name,
                                                            " (component)" if self.is_component else "")

class MemRegion(HsaWrapper):
    """Abstracts a HSA memory region.

    This will wrap and provide an OO interface for hsa_region_t C-API elements
    """
    _hsa_info_function = 'hsa_region_get_info'
    _hsa_properties = {
        'base': (enums.HSA_REGION_INFO_BASE, drvapi.hsa_uintptr_t),
        'size': (enums.HSA_REGION_INFO_SIZE, ctypes.c_size_t),
        '_agent': (enums.HSA_REGION_INFO_AGENT, drvapi.hsa_agent_t),
        '_flags': (enums.HSA_REGION_INFO_FLAGS, drvapi.hsa_region_flag_t),
        'segment': (enums.HSA_REGION_INFO_SEGMENT, drvapi.hsa_segment_t),
        'alloc_max_size': (enums.HSA_REGION_INFO_ALLOC_MAX_SIZE, ctypes.c_size_t),
        'alloc_granule': (enums.HSA_REGION_INFO_ALLOC_GRANULE, ctypes.c_size_t),
        'alloc_alignment': (enums.HSA_REGION_INFO_ALLOC_ALIGNMENT, ctypes.c_size_t),
        'bandwidth': (enums.HSA_REGION_INFO_BANDWIDTH, ctypes.c_uint32),
        'node': (enums.HSA_REGION_INFO_NODE, ctypes.c_uint32),
    }

    def __init__(self, region_id):
        """Do not instantiate MemRegion objects directly, use the factory class
        method 'instance_for' to ensure MemRegion identity"""
        self._id = region_id

    @property
    def agent(self):
        return Agent(self._agent)

    @property
    def supports_kernargs(self):
        return self._flags & enums.HSA_REGION_FLAG_KERNARG

    def allocate(self, ctypes_type):
        buff = ctypes.c_void_p()
        hsa.hsa_memory_allocate(self._id, ctypes.sizeof(ctypes_type), ctypes.byref(buff))
        return ctypes.cast(buff, ctypes.POINTER(ctypes_type)).contents

    def free(self, ptr):
        hsa.hsa_memory_free(ctypes.byref(ptr))

    _instance_dict = {}
    @classmethod
    def instance_for(cls, _id):
        try:
            return cls._instance_dict[_id]
        except KeyError:
            new_instance = cls(_id)
            cls._instance_dict[_id] = new_instance
            return new_instance


class Queue(object):
    def __init__(self, queue_ptr):
        """The id in a queue is a pointer to the queue object returned by hsa_queue_create.
        The Queue object has ownership on that queue object"""
        self._id = queue_ptr


    def __del__(self):
        hsa.hsa_queue_destroy(self._id)
        pass

    def __getattr__(self, fname):
        return getattr(self._id.contents, fname)


    def dispatch(self, code_descriptor, kernargs,
                 workgroup_size=None,
                 grid_size=None,
                 signal=None):
        dims = len(workgroup_size)
        assert dims == len(grid_size)
        assert dims <= 3
        s = signal if signal is not None else hsa.create_signal(1)

        cd = code_descriptor._id

        aql = drvapi.hsa_dispatch_packet_t()
        aql.header.type = enums.HSA_PACKET_TYPE_DISPATCH
        aql.header.acquire_fence_scope = 2
        aql.header.release_fence_scope = 2
        aql.header.barrier = 1
        aql.group_segment_size = 0
        aql.private_segment_size = 0
        aql.dimensions = dims
        aql.workgroup_size_x = workgroup_size[0]
        aql.workgroup_size_y = workgroup_size[1] if dims > 1 else 1
        aql.workgroup_size_z = workgroup_size[2] if dims > 2 else 1
        aql.grid_size_x = grid_size[0]
        aql.grid_size_y = grid_size[1] if dims > 1 else 1
        aql.grid_size_z = grid_size[2] if dims > 2 else 1
        aql.kernel_object_address = cd.code.handle
        aql.kernarg_address = ctypes.cast(ctypes.byref(kernargs), ctypes.c_void_p).value
        aql.completion_signal = s._id

        q = self._id
        index = hsa.hsa_queue_load_write_index_relaxed(q)
        hsa.hsa_queue_store_write_index_relaxed(q, index+1)
        queue_mask = q.contents.size - 1
        real_index = index & queue_mask
        packet_array = ctypes.cast(q.contents.base_address,
                                   ctypes.POINTER(drvapi.hsa_dispatch_packet_t))
        packet_array[real_index] = aql
        hsa.hsa_signal_store_relaxed(self.doorbell_signal, index)

        # synchronous if no signal was provided
        if signal is None:
            hsa.hsa_signal_wait_acquire(s._id, enums.HSA_LT, 1, -1, enums.HSA_WAIT_EXPECTANCY_UNKNOWN)

    def __dir__(self):
        return sorted(set(dir(self._id.contents) +
                          self.__dict__.keys()))


class Signal(object):
    """The id for the signal is going to be the hsa_signal_t returned by create_signal.
    Lifetime of the underlying signal will be tied with this object".
    Note that it is likely signals will have lifetime issues."""
    def __init__(self, signal_id):
        self._id = signal_id

    def __del__(self):
        hsa.hsa_signal_destroy(self._id)



class BrigModule(object):
    def __init__(self, brig_module_id):
        self._id = brig_module_id

    def __del__(self):
        elf_utils.destroy_brig_module(self._id)

    @classmethod
    def from_file(cls, file_name):
        result = ctypes.POINTER(drvapi.hsa_ext_brig_module_t)()
        elf_utils.create_brig_module_from_brig_file(
            file_name.encode('utf8'),
            ctypes.byref(result))
        return BrigModule(result.contents)

    def find_symbol_offset(self, symbol_name):
        symbol_offset = drvapi.hsa_ext_brig_code_section_offset32_t()
        elf_utils.find_symbol_offset(self._id, symbol_name.encode('utf8'),
                                     ctypes.byref(symbol_offset))

        return symbol_offset.value


class Program(object):
    def __init__(self, program_id):
        self._id = program_id

    def __del__(self):
        hsa.hsa_ext_program_destroy(self._id)

    def add_module(self, module):
        result = drvapi.hsa_ext_brig_module_handle_t()
        hsa.hsa_ext_add_module(self._id, module._id, ctypes.byref(result))
        return BrigModuleHandle(result)

    def finalize(self, device, module, symbol,
                 call_convention=0,
                 error_message_callback=None,
                 opt_level=0,
                 options=None,
                 debug_info=0):
        request = drvapi.hsa_ext_finalization_request_t()
        request.module.handle = module._id.handle
        request.program_call_convention = call_convention
        request.symbol = symbol

        cb_typ = drvapi.hsa_ext_error_message_callback_t
        cb = error_message_callback
        cb = ctypes.cast(None, cb_typ) if cb is None else cb_typ(cb)

        hsa.hsa_ext_finalize_program(self._id, device._id, 1,
                                     ctypes.byref(request),
                                     None, # control_directives
                                     cb,
                                     opt_level,
                                     options,
                                     debug_info)

        kernel_id = ctypes.POINTER(drvapi.hsa_ext_code_descriptor_t)()
        hsa.hsa_ext_query_kernel_descriptor_address(
            self._id, module._id, symbol,
            ctypes.byref(kernel_id))

        return CodeDescriptor(kernel_id.contents)

class BrigModuleHandle(object):
    def __init__(self, module_handle_id):
        self._id = module_handle_id

    def __del__(self):
        # this handle seems to be owned by the program, probably valid within that
        # program...
        pass

class CodeDescriptor(object):
    def __init__(self, code_descriptor_id):
        self._id = code_descriptor_id
