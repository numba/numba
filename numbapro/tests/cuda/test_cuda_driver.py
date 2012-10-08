from numbapro._cuda.driver import *

driver = Driver()
print driver.get_device_count()

device = Device(driver, 0)
print device
print device.attributes

context = Context(device)
print context

ptx = '''


.version 3.0
.target sm_20
.address_size 64

	.file	1 "/tmp/tmpxft_000012c7_00000000-9_testcuda.cpp3.i"
	.file	2 "testcuda.cu"

.entry _Z10helloworldPi(
	.param .u64 _Z10helloworldPi_param_0
)
{
	.reg .s32 	%r<3>;
	.reg .s64 	%rl<5>;


	ld.param.u64 	%rl1, [_Z10helloworldPi_param_0];
	cvta.to.global.u64 	%rl2, %rl1;
	.loc 2 6 1
	mov.u32 	%r1, %tid.x;
	mul.wide.u32 	%rl3, %r1, 4;
	add.s64 	%rl4, %rl2, %rl3;
	st.global.u32 	[%rl4], %r1;
	.loc 2 7 2
	ret;
}




'''

module = Module(context, ptx)
print

function = Function(module, '_Z10helloworldPi')
print function

array = (c_int * 100)()
memory = DeviceMemory(context, sizeof(array))
memory.to_device_raw(array, sizeof(array))

print memory

function = function.configure((1,), (100,))
function(memory)

memory.from_device_raw(array, sizeof(array))
for i, v in enumerate(array):
    assert i == v

print 'ok'
