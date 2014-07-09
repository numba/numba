# sample based on Simple Convolution (AMD) sample

from numba.ocl.ocldrv import cl
from numba.ocl.ocldrv.types import *
import numpy as np
from skimage import data, io, filter
from itertools import product as it_product

opencl_program = b"""
__kernel void simpleConvolution(__global  float  * output,
                                __global  float  * input,
                                __global  float  * mask,
                                const     uint2  inputDimensions,
                                const     uint2  maskDimensions)
{
    uint tid   = get_global_id(0);

    uint width  = inputDimensions.x;
    uint height = inputDimensions.y;

    uint x      = tid%width;
    uint y      = tid/width;

    uint maskWidth  = maskDimensions.x;
    uint maskHeight = maskDimensions.y;

    uint vstep = (maskWidth  -1)/2;
    uint hstep = (maskHeight -1)/2;

    /*
     * find the left, right, top and bottom indices such that
     * the indices do not go beyond image boundaires
     */
    uint left    = (x           <  vstep) ? 0         : (x - vstep);
    uint right   = ((x + vstep) >= width) ? width - 1 : (x + vstep);
    uint top     = (y           <  hstep) ? 0         : (y - hstep);
    uint bottom  = ((y + hstep) >= height)? height - 1: (y + hstep);

    /*
     * initializing wighted sum value
     */
    float sumFX = 0;

    for(uint i = left; i <= right; ++i)
        for(uint j = top ; j <= bottom; ++j)
        {
            /*
             * performing wighted sum within the mask boundaries
             */
            uint maskIndex = (j - (y - hstep)) * maskWidth  + (i - (x - vstep));
            uint index     = j                 * width      + i;

            sumFX += (input[index] * mask[maskIndex]);
        }

    output[tid] = clamp(sumFX,0.0f,1.0f);
}
"""

input_image = data.moon()
if np.issubdtype(input_image.dtype, np.integer):
    factor = 1.0/np.iinfo(input_image.dtype).max
    input_image = input_image.astype(np.float32) * factor #dequant

print("input image:\n{0}\n".format(input_image))
output_image = np.empty_like(input_image)

#this makes a fancy mask for convolution
mask = np.array([60.0/25.0 - (abs(x-2) + abs(y-2)) for x,y in it_product(range(5), range(5))], dtype=np.float32).reshape(5,5)

print("Using mask:\n{0}\n".format(mask))

platform = cl.platforms[0]
device = platform.all_devices[-1]
ctxt = cl.create_context(platform, [device])
input_buf = ctxt.create_buffer(input_image.nbytes)
output_buf = ctxt.create_buffer(output_image.nbytes)
mask_buf = ctxt.create_buffer(mask.nbytes)

program = ctxt.create_program_from_source(opencl_program)
program.build()

q = ctxt.create_command_queue(device)
q.enqueue_write_buffer(input_buf, 0, input_image.nbytes, input_image.ctypes.data)
q.enqueue_write_buffer(mask_buf, 0, mask.nbytes, mask.ctypes.data)
kernel = program.create_kernel(b"simpleConvolution")
img_dims = (cl_uint*2)(*input_image.shape)
mask_dims = (cl_uint*2)(*mask.shape)
kernel.set_arg(0, output_buf)
kernel.set_arg(1, input_buf)
kernel.set_arg(2, mask_buf)
kernel.set_arg(3, img_dims)
kernel.set_arg(4, mask_dims)

local_sz = kernel.get_work_group_size_for_device(device)
global_sz = input_image.shape[0]*input_image.shape[1]
q.enqueue_nd_range_kernel(kernel, 1, [global_sz], [local_sz])
q.finish()

q.enqueue_read_buffer(output_buf, 0, output_image.nbytes, output_image.ctypes.data)
print("output image:\n{0}\n".format(output_image))
imgplot = io.imshow(output_image)
io.show()
