from pycuda import driver
from time import time
import numpy as np

def adjust_size(x):
    UNITS = ['B', 'KB', 'MB', 'GB']
    for i, unit in enumerate(UNITS):
        base = 2**(i * 10)
        if x >= base and x < base * (2**10):
            break
    return (x // base), unit

def print_memory_use():
    free, total = driver.mem_get_info()
    args = adjust_size(free) + adjust_size(total)
    print("Memory usage: %d%s free | %d%s total" % args)

    
def main():
    # initialize device
    from pycuda.autoinit import device, context
    print("Running on device: %s" % device.name())
    print("Device memory: %d%s" % adjust_size(device.total_memory()))

    devattr = device.get_attributes()
    MAX_THREAD = devattr[driver.device_attribute.MAX_THREADS_PER_BLOCK]
    MAX_BLOCK = devattr[driver.device_attribute.MAX_BLOCK_DIM_X]
    
    print("Max. threads per block: %d" % MAX_THREAD)
    print("Max. blocks per grid: %d" % MAX_BLOCK)

    with open("add.ptx", "rb") as ptxfile:
        module = driver.module_from_buffer(ptxfile.read())

    kernel_ptx_add = module.get_function("ptx_add")

    print_memory_use()

    # prepare data
    #    at host
    N = MAX_THREAD * MAX_BLOCK
    A = np.arange(N, dtype=np.int32)
    B = np.arange(N, dtype=np.int32)
    S = np.zeros_like(A)

    #    at device
    print("Populate device memory")
    devA = driver.to_device(A)
    devB = driver.to_device(B)
    devS = driver.to_device(S)

    print_memory_use()

    # device compute
    if N > MAX_THREAD:
        threadct =  MAX_THREAD, 1, 1
        blockct = (N / MAX_THREAD), 1
    else:
        threadct =  N, 1, 1
        blockct  =  1, 1

    print("Kernel dimension: threads = %s blocks = %s" % (threadct, blockct))

    dev_time = kernel_ptx_add(devS, devA, devB,
                             block=threadct, grid=blockct,
                             time_kernel=True)
    print("Device completed in %fs" % dev_time)

    # retrieve result
    resS = driver.from_device_like(devS, S)
    print("A + B = %s" % resS)

    # host compute
    ts = time()
    goldS = A + B
    host_time = time() - ts
    
    print("Host completed in %fs" % host_time)

    # check result
    print("Device is %.1fx faster" % (host_time/dev_time))
    
    assert (goldS == resS).all(), "Computation error"
    print("All good")
    
if __name__ == "__main__":
    main()
