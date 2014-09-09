# A script to build external dependencies

import os
import subprocess


NVCC = 'nvcc'
CUB_INCLUDE = '-Ithirdparty/cub'
MGPU_INCLUDE = '-Ithirdparty/moderngpu/include'

GENCODE_SMXX = "-gencode arch=compute_{CC},code=sm_{CC}"
GENCODE_SM20 = GENCODE_SMXX.format(CC=20)
GENCODE_SM30 = GENCODE_SMXX.format(CC=30)
GENCODE_SM35 = GENCODE_SMXX.format(CC=35)

SM = []
SM.append(GENCODE_SM20)
SM.append(GENCODE_SM30)
SM.append(GENCODE_SM35)

GENCODE_FLAGS = ' '.join(SM)


def run_shell(cmd):
    print(cmd)
    subprocess.check_call(cmd, shell=True)


def build_cuda(basepath, out, ins, includes):
    output = os.path.join(basepath, out)
    inputs = ' '.join([os.path.join(basepath, p)
                       for p in ins])
    argtemp = '-m64 --compiler-options "-fPIC" {} -O3 {} --shared -o {} {}'
    args = argtemp.format(includes, GENCODE_FLAGS, output, inputs)
    cmd = ' '.join([NVCC, args])
    run_shell(cmd)


def build_radixsort():
    build_cuda(basepath='numbapro/cudalib/sorting/details/',
               out='radixsort.so',
               ins=['cubradixsort.cu'],
               includes=CUB_INCLUDE,)


def build_mgpusort():
    build_cuda(basepath='numbapro/cudalib/sorting/details/',
               out='mgpusort.so',
               ins=['mgpusort.cu'],
               includes=MGPU_INCLUDE,)


if __name__ == '__main__':
    build_radixsort()
    build_mgpusort()
