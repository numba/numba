from __future__ import print_function, absolute_import
import os
import ctypes

import numpy as np

import numba.unittest_support as unittest
from numba.hsa.hsadrv.driver import hsa, Queue, BrigModule
from numba.hsa.hsadrv import drvapi, enums


class TestAgents(unittest.TestCase):
    def test_agents_init(self):
        self.assertGreater(len(hsa.agents), 0)

    def test_agents_create_queue_single(self):
        for agent in hsa.agents:
            queue = agent.create_queue_single(2 ** 5)
            self.assertIsInstance(queue, Queue)

    def test_agents_create_queue_multi(self):
        for agent in hsa.agents:
            queue = agent.create_queue_multi(2 ** 5)
            self.assertIsInstance(queue, Queue)


class _TestBase(unittest.TestCase):
    def setUp(self):
        self.gpu = [a for a in hsa.agents if a.is_component][0]
        self.cpu = [a for a in hsa.agents if not a.is_component][0]
        self.queue = self.gpu.create_queue_multi(self.gpu.queue_max_size)

    def tearDown(self):
        del self.queue
        del self.gpu
        del self.cpu


def get_brig_file():
    basedir = os.path.dirname(__file__)
    path = os.path.join(basedir, 'vector_copy.brig')
    assert os.path.isfile(path)
    return path


class TestBrigModule(unittest.TestCase):
    def test_from_file(self):
        brig_file = get_brig_file()
        brig_module = BrigModule.from_file(brig_file)
        offset = brig_module.find_symbol_offset('&__vector_copy_kernel')
        self.assertNotEqual(offset, 0)


class TestProgram(_TestBase):
    def test_create_program(self):
        brig_file = get_brig_file()
        symbol = '&__vector_copy_kernel'
        brig_module = BrigModule.from_file(brig_file)
        symbol_offset = brig_module.find_symbol_offset(symbol)
        program = hsa.create_program([self.gpu])
        module = program.add_module(brig_module)
        code_descriptor = program.finalize(self.gpu, module, symbol_offset)
        self.assertGreater(code_descriptor._id.kernarg_segment_byte_size, 0)


class TestMemory(_TestBase):
    def test_register(self):
        src = np.random.random(1024).astype(np.float32)
        hsa.hsa_memory_register(src.ctypes.data, src.nbytes)
        hsa.hsa_memory_deregister(src.ctypes.data, src.nbytes)

    def test_allocate(self):
        my_region = drvapi.hsa_region_t(0)

        def _callback(region, result):
            flags = drvapi.hsa_region_flag_t()
            hsa.hsa_region_get_info(region, enums.HSA_REGION_INFO_FLAGS,
                                    ctypes.byref(flags))
            if not (flags.value & enums.HSA_REGION_FLAG_KERNARG):
                result.value = region
            return enums.HSA_STATUS_SUCCESS

        callback = drvapi.HSA_AGENT_ITERATE_REGIONS_CALLBACK_FUNC(_callback)
        hsa.hsa_agent_iterate_regions(self.cpu._id, callback, my_region)

        self.assertNotEqual(my_region.value, 0)

        src = np.random.random(1024).astype(np.float32)

        ptr = ctypes.c_void_p(0)
        hsa.hsa_memory_allocate(my_region, src.nbytes, ctypes.byref(ptr))
        ctypes.memmove(ptr, src.ctypes.data, src.nbytes)

        data = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float))

        for i in range(src.size):
            self.assertEqual(data[i], src[i])

        hsa.hsa_memory_free(ptr)


if __name__ == '__main__':
    unittest.main()
