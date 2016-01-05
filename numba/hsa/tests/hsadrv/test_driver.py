from __future__ import print_function, absolute_import
import os
import ctypes

import numpy as np

import numba.unittest_support as unittest
from numba.hsa.hsadrv.driver import hsa, Queue, Program, Executable,\
                                    BrigModule, Context
from numba.hsa.hsadrv import drvapi
from numba.hsa.hsadrv import enums
from numba.hsa.hsadrv import enums_ext

from numba import config

# hardware
known_dgpus = frozenset([b'Fiji'])
known_apus = frozenset([b'Spectre'])
known_cpus = frozenset([b'Kaveri'])


def apu_present():
    """
    Returns true if an APU is present on the current machine.
    """
    # Find the nodes to which the agents claim to belong.
    # If the number of nodes is different to the number of
    # agents then some agents must share a node -> APU!
    nodes = set()
    for a in hsa.agents:
        nodes.add(getattr(a, "node"))
    return len(hsa.agents) != len(nodes)


def dgpu_count():
    """
    Returns the number of discrete GPUs present on the current machine.

    This can be overridden by setting the environment variable
    `NUMBA_HSA_DGPU_PRESENT` to a positive integer.
    """
    if config.NUMBA_HSA_DGPU_PRESENT:
        return config.NUMBA_HSA_DGPU_PRESENT
    else:
        ngpus = 0
        for a in hsa.agents:
            if a.is_component:
                name = getattr(a, "name").lower()
                for g in known_dgpus:
                    if g.lower() in name:
                        ngpus += 1
        return ngpus

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
        kernarg_regions = list()
        for r in regions:
            if r.supports(enums.HSA_REGION_GLOBAL_FLAG_KERNARG):
                kernarg_regions.append(r)

        self.assertGreater(len(kernarg_regions), 0)
        # Test allocating at the kernel argument region
        kernarg_region = kernarg_regions[0]
        nelem = 10
        ptr = kernarg_region.allocate(ctypes.sizeof(ctypes.c_float) * nelem)
        self.assertNotEqual(ctypes.addressof(ptr), 0,
                            "pointer must not be NULL")
        # Test writing to it
        src = np.random.random(nelem).astype(np.float32)
        ctypes.memmove(ptr, src.ctypes.data, src.nbytes)

        ref = (ctypes.c_float * nelem).from_address(ptr.value)
        for i in range(src.size):
            self.assertEqual(ref[i], src[i])
        hsa.hsa_memory_free(ptr)

    @unittest.skipIf(dgpu_count() == 0, "no discrete GPU present")
    def test_coarse_grained_allocate(self):
        """
        Tests the coarse grained allocation works on a dGPU.
        It performs a data copying round trip via:
        memory
          |
        HSA cpu memory
          |
        HSA dGPU host accessible memory <---|
          |                                 |
        HSA dGPU memory --------------------|
        """
        gpu_regions = self.gpu.regions
        gpu_only_coarse_regions = list()
        gpu_host_accessible_coarse_regions = list()
        for r in gpu_regions:
            if r.supports(enums.HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED):
                if r.host_accessible:
                    gpu_host_accessible_coarse_regions.append(r)
                else:
                    gpu_only_coarse_regions.append(r)

        # check we have 1+ coarse gpu region(s) of each type
        self.assertGreater(len(gpu_only_coarse_regions), 0)
        self.assertGreater(len(gpu_host_accessible_coarse_regions), 0)

        cpu_regions = self.cpu.regions
        cpu_coarse_regions = list()
        for r in cpu_regions:
            if r.supports(enums.HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED):
                cpu_coarse_regions.append(r)
        # check we have 1+ coarse cpu region(s)
        self.assertGreater(len(cpu_coarse_regions), 0)

        # ten elements of data used
        nelem = 10

        # allocation
        cpu_region = cpu_coarse_regions[0]
        cpu_ptr = cpu_region.allocate(ctypes.sizeof(ctypes.c_float) * nelem)
        self.assertNotEqual(ctypes.addressof(cpu_ptr), 0,
                "pointer must not be NULL")

        gpu_only_region = gpu_only_coarse_regions[0]
        gpu_only_ptr = gpu_only_region.allocate(ctypes.sizeof(ctypes.c_float) *
                nelem)
        self.assertNotEqual(ctypes.addressof(gpu_only_ptr), 0,
                "pointer must not be NULL")

        gpu_host_accessible_region = gpu_host_accessible_coarse_regions[0]
        gpu_host_accessible_ptr = gpu_host_accessible_region.allocate(
                ctypes.sizeof(ctypes.c_float) * nelem)
        self.assertNotEqual(ctypes.addressof(gpu_host_accessible_ptr), 0,
                "pointer must not be NULL")

        # Test writing to allocated area
        src = np.random.random(nelem).astype(np.float32)
        hsa.hsa_memory_copy(cpu_ptr, src.ctypes.data, src.nbytes)
        hsa.hsa_memory_copy(gpu_host_accessible_ptr, cpu_ptr, src.nbytes)
        hsa.hsa_memory_copy(gpu_only_ptr, gpu_host_accessible_ptr, src.nbytes)

        # check write is correct
        cpu_ref = (ctypes.c_float * nelem).from_address(cpu_ptr.value)
        for i in range(src.size):
            self.assertEqual(cpu_ref[i], src[i])

        gpu_ha_ref = (ctypes.c_float * nelem).\
            from_address(gpu_host_accessible_ptr.value)
        for i in range(src.size):
            self.assertEqual(gpu_ha_ref[i], src[i])

        # zero out host accessible GPU memory and CPU memory
        z0 = np.zeros(nelem).astype(np.float32)
        hsa.hsa_memory_copy(cpu_ptr, z0.ctypes.data, z0.nbytes)
        hsa.hsa_memory_copy(gpu_host_accessible_ptr, cpu_ptr, z0.nbytes)

        # check zeroing is correct
        for i in range(z0.size):
            self.assertEqual(cpu_ref[i], z0[i])

        for i in range(z0.size):
            self.assertEqual(gpu_ha_ref[i], z0[i])

        # copy back the data from the GPU
        hsa.hsa_memory_copy(gpu_host_accessible_ptr, gpu_only_ptr, src.nbytes)

        # check the copy back is ok
        for i in range(src.size):
            self.assertEqual(gpu_ha_ref[i], src[i])

        # free
        hsa.hsa_memory_free(cpu_ptr)
        hsa.hsa_memory_free(gpu_only_ptr)
        hsa.hsa_memory_free(gpu_host_accessible_ptr)

    @unittest.skipIf(dgpu_count() == 0, "no discrete GPU present")
    def test_coarse_grained_kernel_execution(self):
        """
        This tests the execution of a kernel on a dGPU using coarse memory
        regions for the buffers.
        NOTE: the code violates the HSA spec in that it uses a coarse region
        for kernargs, this is a performance hack.
        """

        from numba.hsa.hsadrv.driver import BrigModule, Program, hsa,\
                Executable

        def get_brig_file(basedir=None):
            if basedir == None:
                basedir = os.path.dirname(__file__)
            path = os.path.join(basedir, 'vector_copy_dgpu_example.brig')
            assert os.path.isfile(path)
            return path

        # get a brig file
        brig_file = get_brig_file('/srv/data/hsa')
        brig_module = BrigModule.from_file(brig_file)
        self.assertGreater(len(brig_module), 0)

        # use existing GPU regions for computation space
        gpu_regions = self.gpu.regions
        gpu_only_coarse_regions = list()
        gpu_host_accessible_coarse_regions = list()
        for r in gpu_regions:
            if r.supports(enums.HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED):
                if r.host_accessible:
                    gpu_host_accessible_coarse_regions.append(r)
                else:
                    gpu_only_coarse_regions.append(r)

        # check we have 1+ coarse gpu region(s) of each type
        self.assertGreater(len(gpu_only_coarse_regions), 0)
        self.assertGreater(len(gpu_host_accessible_coarse_regions), 0)

        # Compilation phase:

        # FIXME: this is dubious, assume launching agent is indexed first
        agent = hsa.components[0]

        # dGPU needs base profile, not full
        prog = Program(profile=enums.HSA_PROFILE_BASE)
        prog.add_module(brig_module)

        # get kernel and load
        code = prog.finalize(agent.isa)

        ex = Executable()
        ex.load(agent, code)
        ex.freeze()

        # extract symbols
        sym = ex.get_symbol(agent, "&__vector_copy_kernel")
        self.assertNotEqual(sym.kernel_object, 0)
        self.assertGreater(sym.kernarg_segment_size, 0)

        # attempt kernel excution
        import ctypes
        import numpy as np

        # Do memory allocations

        # allocate and initialise memory
        nelem = 1024 * 1024

        src = np.random.random(nelem).astype(np.float32)
        z0 = np.zeros_like(src)

        # alloc host accessible memory
        nbytes = ctypes.sizeof(ctypes.c_float) * nelem
        gpu_host_accessible_region = gpu_host_accessible_coarse_regions[0]
        host_in_ptr = gpu_host_accessible_region.allocate(nbytes)
        self.assertNotEqual(host_in_ptr.value, None,
                "pointer must not be NULL")
        host_out_ptr = gpu_host_accessible_region.allocate(nbytes)
        self.assertNotEqual(host_out_ptr.value, None,
                "pointer must not be NULL")

        # init mem with data
        hsa.hsa_memory_copy(host_in_ptr, src.ctypes.data, src.nbytes)
        hsa.hsa_memory_copy(host_out_ptr, z0.ctypes.data, z0.nbytes)

        # alloc gpu only memory
        gpu_only_region = gpu_only_coarse_regions[0]
        gpu_in_ptr = gpu_only_region.allocate(nbytes)
        self.assertNotEqual(gpu_in_ptr.value, None, "pointer must not be NULL")
        gpu_out_ptr = gpu_only_region.allocate(nbytes)
        self.assertNotEqual(gpu_out_ptr.value, None,
            "pointer must not be NULL")

        # copy memory from host accessible location to gpu only
        hsa.hsa_memory_copy(gpu_in_ptr, host_in_ptr, src.nbytes)

        # Do kernargs

        # Find a coarse region (for better performance on dGPU) in which
        # to place kernargs. NOTE: This violates the HSA spec
        kernarg_regions = list()
        for r in gpu_host_accessible_coarse_regions:
           # NOTE: VIOLATION
           # if r.supports(enums.HSA_REGION_GLOBAL_FLAG_KERNARG):
           kernarg_regions.append(r)
        self.assertGreater(len(kernarg_regions), 0)

        # use first region for args
        kernarg_region = kernarg_regions[0]

        kernarg_ptr = kernarg_region.allocate(
                2 * ctypes.sizeof(ctypes.c_void_p))

        self.assertNotEqual(kernarg_ptr, None, "pointer must not be NULL")

        # wire in gpu memory
        argref = (2 * ctypes.c_size_t).from_address(kernarg_ptr.value)
        argref[0] = gpu_in_ptr.value
        argref[1] = gpu_out_ptr.value
        kernarg_ptr = argref

        # signal
        sig = hsa.create_signal(1)

        # create queue and dispatch job

        queue = agent.create_queue_single(32)
        queue.dispatch(sym, kernarg_ptr, workgroup_size=(256, 1, 1),
                           grid_size=(nelem, 1, 1),signal=None)

        # copy result back to host accessible memory to check
        hsa.hsa_memory_copy(host_out_ptr, gpu_out_ptr, src.nbytes)

        # check the data is recovered
        ref = (nelem * ctypes.c_float).from_address(host_out_ptr.value)
        np.testing.assert_equal(ref, src)

        # free
        hsa.hsa_memory_free(host_in_ptr)
        hsa.hsa_memory_free(host_out_ptr)
        hsa.hsa_memory_free(gpu_in_ptr)
        hsa.hsa_memory_free(gpu_out_ptr)

class TestContext(_TestBase):
    """Tests the Context class behaviour is correct."""

    def test_memalloc(self):
        """
            Tests Context.memalloc() for a given, in the parlance of HSA,\
            `component`. Testing includes specialisations for the supported
            components of dGPUs and APUs.
        """
        n = 10 # things to alloc
        nbytes = ctypes.sizeof(ctypes.c_double) * n

        # run if a dGPU is present
        if(dgpu_count() > 0):
            # find a host accessible region
            dGPU_agent = self.gpu
            CPU_agent = self.cpu
            gpu_ctx = Context(dGPU_agent)
            gpu_only_mem = gpu_ctx.memalloc(nbytes, hostAccessible=False)
            ha_mem = gpu_ctx.memalloc(nbytes, hostAccessible=True)

            # on dGPU systems, all host mem is host accessible
            cpu_ctx = Context(CPU_agent)
            cpu_mem = cpu_ctx.memalloc(nbytes, hostAccessible=True)

            # Test writing to allocated area
            src = np.random.random(n).astype(np.float64)
            hsa.hsa_memory_copy(cpu_mem.device_pointer, src.ctypes.data, src.nbytes)
            hsa.hsa_memory_copy(ha_mem.device_pointer, cpu_mem.device_pointer, src.nbytes)
            hsa.hsa_memory_copy(gpu_only_mem.device_pointer, ha_mem.device_pointer, src.nbytes)

            # clear

        else: #TODO: write APU variant
            pass



if __name__ == '__main__':
    unittest.main()
