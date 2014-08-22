# A script to build external dependencies

import os
import subprocess


NVCC = 'nvcc'
CUB_INCLUDE = '-Ithirdparty/cub'

GENCODE_SMXX = "-gencode arch=compute_{CC},code=sm_{CC}"
GENCODE_SM20 = GENCODE_SMXX.format(CC=20)
GENCODE_SM30 = GENCODE_SMXX.format(CC=30)
GENCODE_SM35 = GENCODE_SMXX.format(CC=35)

SM = []
# SM.append(GENCODE_SM20)
SM.append(GENCODE_SM30)
SM.append(GENCODE_SM35)

GENCODE_FLAGS = ' '.join(SM)


def run_shell(cmd):
    print(cmd)
    subprocess.check_call(cmd, shell=True)


def build_radixsort():
    basepath = 'numbapro/cudalib/sorting/details/'
    output = os.path.join(basepath, 'radixsort.so')
    inputs = os.path.join(basepath, 'cubradixsort.cu')
    argtemp = '-m64 --compiler-options "-fPIC" {} -O3 {} --shared -o {} {}'
    args = argtemp.format(CUB_INCLUDE, GENCODE_FLAGS, output, inputs)
    cmd = ' '.join([NVCC, args])
    run_shell(cmd)


if __name__ == '__main__':
    build_radixsort()
