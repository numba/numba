/* OpenCl kernel for element-wise addition of two arrays */
kernel void oneapiPy_oneapi_py_devfn__5F__5F_main_5F__5F__2E_data_5F_parallel_5F_sum_24_1_2E_array_28_float32_2C__20_1d_2C__20_C_29__2E_array_28_float32_2C__20_1d_2C__20_C_29__2E_array_28_float32_2C__20_1d_2C__20_C_29_(
    void *a_meminfo, global void *a_parent, unsigned int a_size,
    unsigned int a_itemsize, global float *a, unsigned int a_shape_d0, unsigned int a_stride_d0,
    global void *b_meminfo, global void *b_parent, unsigned int b_size,
    unsigned int b_itemsize, global float *b, unsigned int b_shape_d0, unsigned int b_stride_d0,
    global void *c_meminfo, global void *c_parent, unsigned int c_size,
    unsigned int c_itemsize, global float *c, unsigned int c_shape_d0, unsigned int c_stride_d0
    )
{
    const int idx = get_global_id(0);
    c[idx] = a[idx] + b[idx];
}