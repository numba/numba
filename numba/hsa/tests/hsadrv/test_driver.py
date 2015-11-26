from __future__ import print_function, absolute_import
import os
import ctypes

import numpy as np

import numba.unittest_support as unittest
from numba.hsa.hsadrv.driver import hsa, Queue, Program, Executable, BrigModule
from numba.hsa.hsadrv import drvapi


class TestLowLevelApi(unittest.TestCase):
    """This test checks that all the functions defined in drvapi
    bind properly using ctypes."""

    def test_functions_available(self):
        missing_functions = []
        for fname in drvapi.API_PROTOTYPES.keys():
            try:
                getattr(hsa, fname)
            except Exception as e:
                missing_functions.append("'{0}': {1}".format(fname, str(e)))

        self.assertEqual(len(missing_functions), 0,
                         msg='\n'.join(missing_functions))


class TestAgents(unittest.TestCase):
    def test_agents_init(self):
        self.assertGreater(len(hsa.agents), 0)

    def test_agents_create_queue_single(self):
        for agent in hsa.agents:
            if agent.is_component:
                queue = agent.create_queue_single(2 ** 5)
                self.assertIsInstance(queue, Queue)

    def test_agents_create_queue_multi(self):
        for agent in hsa.agents:
            if agent.is_component:
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
        self.assertGreater(len(brig_module), 0)


class TestProgram(_TestBase):
    def test_create_program(self):
        brig_file = get_brig_file()
        symbol = '&__vector_copy_kernel'
        brig_module = BrigModule.from_file(brig_file)

        program = Program()
        program.add_module(brig_module)
        code = program.finalize(self.gpu.isa)

        ex = Executable()
        ex.load(self.gpu, code)
        ex.freeze()

        sym = ex.get_symbol(self.gpu, symbol)
        self.assertGreater(sym.kernarg_segment_size, 0)


class TestMemory(_TestBase):
    def test_region_list(self):
        self.assertGreater(len(self.gpu.regions.globals), 0)
        self.assertGreater(len(self.gpu.regions.groups), 0)
        # The following maybe empty
        # print(self.gpu.regions.privates)
        # print(self.gpu.regions.readonlys)

    def test_register(self):
        src = np.random.random(1024).astype(np.float32)
        hsa.hsa_memory_register(src.ctypes.data, src.nbytes)
        hsa.hsa_memory_deregister(src.ctypes.data, src.nbytes)

    def test_allocate(self):
        regions = self.gpu.regions
        # More than one region
        self.assertGreater(len(regions), 0)
        # Find kernel argument regions
        kernarg_regions = [r for r in regions if r.supports_kernargs]
        self.assertGreater(len(kernarg_regions), 0)
        # Test allocating at the kernel argument region
        kernarg_resgion = kernarg_regions[0]
        nelem = 10
        ptr = kernarg_resgion.allocate(ctypes.c_float * nelem)
        self.assertNotEqual(ctypes.addressof(ptr), 0,
                            "pointer must not be NULL")
        # # Test writing to it
        src = np.random.random(nelem).astype(np.float32)
        ctypes.memmove(ptr, src.ctypes.data, src.nbytes)
        for i in range(src.size):
            self.assertEqual(ptr[i], src[i])
        hsa.hsa_memory_free(ptr)


if __name__ == '__main__':
    unittest.main()
