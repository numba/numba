from numba.cuda.decorators import declare_device

hsin = declare_device('hsin_wrapper', 'float16(float16)')
hcos = declare_device('hcos_wrapper', 'float16(float16)')
hlog = declare_device('hlog_wrapper', 'float16(float16)')
hlog2 = declare_device('hlog2_wrapper', 'float16(float16)')
hlog10 = declare_device('hlog10_wrapper', 'float16(float16)')
hexp = declare_device('hexp_wrapper', 'float16(float16)')
hexp2 = declare_device('hexp2_wrapper', 'float16(float16)')
hexp10 = declare_device('hexp10_wrapper', 'float16(float16)')
hsqrt = declare_device('hsqrt_wrapper', 'float16(float16)')
hrsqrt = declare_device('hrsqrt_wrapper', 'float16(float16)')
hceil = declare_device('hceil_wrapper', 'float16(float16)')
hfloor = declare_device('hfloor_wrapper', 'float16(float16)')
hrcp = declare_device('hrcp_wrapper', 'float16(float16)')
htrunc = declare_device('htrunc_wrapper', 'float16(float16)')
hrint = declare_device('hrint_wrapper', 'float16(float16)')


