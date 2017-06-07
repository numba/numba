from __future__ import print_function, absolute_import, division

import numba.ocl.ocldrv as ocldrv
from numba.ocl.ocldrv import driver as cl
import numba.unittest_support as unittest

import numpy as np


class TestDeviceProperties(unittest.TestCase):
    def test_device_querying(self):
        all_devices = cl.default_platform.all_devices
        self.assertIn(cl.default_platform.default_device, all_devices)
        for d in cl.default_platform.gpu_devices:
            self.assertIn(d, all_devices)
            d.type_str == 'GPU'
        for d in cl.default_platform.cpu_devices:
            self.assertIn(d, all_devices)
            d.type_str == 'CPU'
        for d in cl.default_platform.accelerator_devices:
            self.assertIn(d, all_devices)
            d.type_str == 'ACCELERATOR'
        self.assertIn(cl.default_platform.default_device, all_devices)

class TestContextProperties(unittest.TestCase):
    def test_default_properties(self):
        device = cl.default_platform.default_device
        context = cl.create_context(device.platform, [device])
        self.assertEqual(context.devices, [device])
        self.assertEqual(context.platform, device.platform)
        self.assertNotEqual(context.reference_count, 0)


class TestQueueProperties(unittest.TestCase):
    def setUp(self):
        self.assertTrue(len(cl.default_platform.all_devices) > 0)
        self.device = cl.default_platform.default_device
        self.context = cl.create_context(self.device.platform, [self.device])

    def tearDown(self):
        del self.context
        del self.device

    def test_default_properties(self):
        q = self.context.create_command_queue(self.device)
        self.assertEqual(q.context, self.context)
        self.assertEqual(q.device, self.device)
        self.assertNotEqual(q.reference_count, 0)
        self.assertEqual(q.properties & ocldrv.CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, 0)
        self.assertEqual(q.properties & ocldrv.CL_QUEUE_PROFILING_ENABLE, 0)


class TestProgramProperties(unittest.TestCase):
    def setUp(self):
        self.assertTrue(len(cl.default_platform.all_devices) > 0)
        self.device = cl.default_platform.default_device
        self.context = cl.create_context(self.device.platform, [self.device])

        self.opencl_source = b"""
__kernel void square(__global float* input, __global float* output, const unsigned int count)
{
    int i = get_global_id(0);
    if (i < count)
        output[i] = input[i] * input[i];
}
"""
    def tearDown(self):
        del self.opencl_source
        del self.context
        del self.device

    def test_simple_program(self):
        program = self.context.create_program_from_source(self.opencl_source)
        program.build()
        self.assertEqual(program.kernel_names, ['square'])
        self.assertEqual(program.devices, [self.device])
        self.assertEqual(program.context, self.context)
        self.assertNotEqual(program.reference_count, 0)
        self.assertEqual(len(program.devices), len(program.binaries))
        self.assertEqual(program.source, self.opencl_source)

        kernel = program.create_kernel(b'square')
        self.assertNotEqual(kernel.reference_count, 0)
        self.assertEqual(kernel.context, self.context)
        self.assertEqual(kernel.program, program)
        self.assertEqual(kernel.num_args, 3)


class TestOpenCLDriver(unittest.TestCase):
    def setUp(self):
        self.assertTrue(len(cl.default_platform.all_devices) > 0)
        self.device = cl.default_platform.cpu_devices[0]
        self.context = cl.create_context(self.device.platform,
                                         [self.device])

        self.opencl_source = b"""
__kernel void square(__global float* input, __global float* output, const unsigned int count)
{
    int i = get_global_id(0);
    if (i < count)
        output[i] = input[i] * input[i];
}
"""
        self.kernel_name = b"square"
        self.DATA_SIZE = 4096
        self.data = np.random.random_sample(self.DATA_SIZE).astype(np.float32)
        self.result = np.empty_like(self.data)

    def tearDown(self):
        del self.result
        del self.data
        del self.DATA_SIZE
        del self.kernel_name
        del self.opencl_source
        del self.context
        del self.device

    def test_ocl_driver_basic(self):
        q = self.context.create_command_queue(self.device)
        buff_in = self.context.create_buffer(self.data.nbytes)
        buff_out = self.context.create_buffer(self.result.nbytes)
        q.enqueue_write_buffer(buff_in, 0, self.data.nbytes, self.data.ctypes.data)
        program = self.context.create_program_from_source(self.opencl_source)
        program.build()
        kernel = program.create_kernel(self.kernel_name)
        kernel.set_arg(0, buff_in)
        kernel.set_arg(1, buff_out)
        kernel.set_arg(2, self.DATA_SIZE)
        local_sz = kernel.get_work_group_size_for_device(self.device)
        global_sz = self.DATA_SIZE
        q.enqueue_nd_range_kernel(kernel, 1, [global_sz], [local_sz])
        q.finish()
        q.enqueue_read_buffer(buff_out, 0, self.result.nbytes, self.result.ctypes.data)
        self.assertEqual(np.sum(self.result == self.data*self.data), self.DATA_SIZE)

    def test_ocl_driver_kernelargs(self):
        q = self.context.create_command_queue(self.device)
        buff_in = self.context.create_buffer(self.data.nbytes)
        buff_out = self.context.create_buffer(self.result.nbytes)
        q.enqueue_write_buffer(buff_in, 0, self.data.nbytes, self.data.ctypes.data)
        program = self.context.create_program_from_source(self.opencl_source)
        program.build()
        kernel = program.create_kernel(self.kernel_name, [buff_in, buff_out, self.DATA_SIZE])
        local_sz = kernel.get_work_group_size_for_device(self.device)
        global_sz = self.DATA_SIZE
        q.enqueue_nd_range_kernel(kernel, 1, [global_sz], [local_sz])
        q.finish()
        q.enqueue_read_buffer(buff_out, 0, self.result.nbytes, self.result.ctypes.data)
        self.assertEqual(np.sum(self.result == self.data*self.data), self.DATA_SIZE)

    def test_ocl_driver_setargs(self):
        q = self.context.create_command_queue(self.device)
        buff_in = self.context.create_buffer(self.data.nbytes)
        buff_out = self.context.create_buffer(self.result.nbytes)
        q.enqueue_write_buffer(buff_in, 0, self.data.nbytes, self.data.ctypes.data)
        program = self.context.create_program_from_source(self.opencl_source)
        program.build()
        kernel = program.create_kernel(self.kernel_name)
        kernel.set_args([buff_in, buff_out, self.DATA_SIZE])
        local_sz = kernel.get_work_group_size_for_device(self.device)
        global_sz = self.DATA_SIZE
        q.enqueue_nd_range_kernel(kernel, 1, [global_sz], [local_sz])
        q.finish()
        q.enqueue_read_buffer(buff_out, 0, self.result.nbytes, self.result.ctypes.data)
        self.assertEqual(np.sum(self.result == self.data*self.data), self.DATA_SIZE)

    def test_ocl_queue_events(self):
        q = self.context.create_command_queue(self.device)
        buff_in = self.context.create_buffer(self.data.nbytes)
        buff_out = self.context.create_buffer(self.result.nbytes)
        evnt = q.enqueue_write_buffer(buff_in, 0, self.data.nbytes, self.data.ctypes.data,
                                      blocking=False, wants_event=True)
        program = self.context.create_program_from_source(self.opencl_source)
        program.build()
        kernel = program.create_kernel(self.kernel_name)
        kernel.set_args([buff_in, buff_out, self.DATA_SIZE])
        local_sz = kernel.get_work_group_size_for_device(self.device)
        global_sz = self.DATA_SIZE
        evnt = q.enqueue_nd_range_kernel(kernel, 1, [global_sz], [local_sz],
                                         wait_list=[evnt], wants_event=True)
        evnt = q.enqueue_read_buffer(buff_out, 0, self.result.nbytes, self.result.ctypes.data,
                                     blocking=False, wait_list=[evnt], wants_event=True)
        cl.wait_for_events(evnt)
        self.assertEqual(np.sum(self.result == self.data*self.data), self.DATA_SIZE)

if __name__ == '__main__':
    unittest.main()