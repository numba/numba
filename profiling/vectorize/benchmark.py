from __future__ import division
import multiprocessing
import numpy as np
from timeit import default_timer as timer
from matplotlib import pyplot
from numbapro import jit, cuda, vectorize

NCPU = multiprocessing.cpu_count()

def checkwork(A, B, C):
    assert np.allclose(C, (A + B) ** 2)

@vectorize(['float32(float32, float32)'], target='cpu')
def cpu_vwork(a, b):
    return (a + b) ** 2

@vectorize(['float32(float32, float32)'], target='parallel')
def par_vwork(a, b):
    return (a + b) ** 2

@vectorize(['float32(float32, float32)'], target='gpu')
def gpu_vwork(a, b):
    return (a + b) ** 2
        
def cpu_run(A, B, C):
    cpu_vwork(A, B, out=C)

def par_run(A, B, C):
    par_vwork(A, B, out=C)

def gpu_run(A, B, C):
    chunksize = 2 ** 25         # GPU RAM dependent
    nsect = C.size // min(chunksize, C.size)
    itr = zip(np.array_split(A, nsect), np.array_split(B, nsect), np.array_split(C, nsect))
    for As, Bs, Cs in itr:
        dA = cuda.to_device(As)
        dB = cuda.to_device(Bs)
        dC = cuda.device_array_like(Cs)
        gpu_vwork(dA, dB, out=dC)
        dC.copy_to_host(Cs)


def main():
    data_sizes = []
    cpu_perf = []
    par_perf = []
    gpu_perf = []
    gpu_stm_perf = []

    BASE = 2
    for power in range(8, 27):
        datasize = BASE ** power
        bytesize = datasize * np.dtype(np.float32).itemsize
        mbsize = bytesize / 2 ** 20
        print 'output data %.3f MB' % (mbsize,)
        A = np.arange(datasize, dtype=np.float32)
        B = np.arange(datasize, dtype=np.float32)
        C = np.zeros(datasize, dtype=np.float32)

        ts = timer()
        cpu_run(A, B, C)
        te = timer()
        cpu_time = te - ts
        checkwork(A, B, C)

        ts = timer()
        par_run(A, B, C)
        te = timer()
        par_time = te - ts
        checkwork(A, B, C)

        ts = timer()
        gpu_run(A, B, C)
        te = timer()
        gpu_time = te - ts
        checkwork(A, B, C)


        data_sizes.append(datasize)
        cpu_perf.append(cpu_time)
        par_perf.append(par_time)
        gpu_perf.append(gpu_time)


    fig, (topplot, botplot) = pyplot.subplots(2, sharex=True)
    topplot.set_xscale('log')
    topplot.set_yscale('log')
    topplot.set_ylabel('duration (seconds in log scale)')
    topplot.plot(data_sizes, cpu_perf, label='CPU-1core')
    topplot.plot(data_sizes, par_perf, label='CPU-%dcore' % NCPU)
    topplot.plot(data_sizes, gpu_perf, label='GPU')

    topplot.legend(loc='upper left')

    botplot.set_xscale('log')
    #botplot.set_yscale('log')
 
    cpu_throughput = np.array(data_sizes) / np.array(cpu_perf)
    par_throughput = np.array(data_sizes) / np.array(par_perf)
    gpu_throughput = np.array(data_sizes) / np.array(gpu_perf)

    botplot.set_ylabel('throughput (float per second)')
    botplot.plot(data_sizes, cpu_throughput, label='CPU-1core')
    botplot.plot(data_sizes, par_throughput, label='CPU-%dcore' % NCPU)
    botplot.plot(data_sizes, gpu_throughput, label='GPU')

    botplot.legend(loc='upper left')

    pyplot.show()
                

if __name__ == '__main__':
    main()

