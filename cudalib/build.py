# A script to build external dependencies

import os
import subprocess


NVCC = os.environ.get('NVCC', 'nvcc')

if tuple.__itemsize__ == 4:
    OPT = '-m32 --compiler-options "-fPIC"'
elif tuple.__itemsize__ == 8:
    OPT = '-m64 --compiler-options "-fPIC"'

CUB_INCLUDE = '-I../thirdparty/cub'
MGPU_INCLUDE = '-I../thirdparty/moderngpu/include'

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


def build_cuda(srcdir, out, ins, includes):
    output = os.path.join(out)
    inputs = ' '.join([os.path.join(srcdir, p)
                       for p in ins])
    argtemp = '{opt} {inc} -O3 {gen} --shared -o {out} {inp}'
    args = argtemp.format(inc=includes, gen=GENCODE_FLAGS, out=output,
                          inp=inputs, opt=OPT)
    cmd = ' '.join([NVCC, args])
    run_shell(cmd)


def build_radixsort():
    build_cuda(srcdir='src',
               out='lib/nbpro_radixsort.so',
               ins=['cubradixsort.cu'],
               includes=CUB_INCLUDE, )


def build_mgpusort():
    build_cuda(srcdir='src',
               out='lib/nbpro_segsort.so',
               ins=['mgpusort.cu'],
               includes=MGPU_INCLUDE, )


if __name__ == '__main__':
    build_radixsort()
    build_mgpusort()
