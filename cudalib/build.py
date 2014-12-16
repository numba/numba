# A script to build external dependencies

import os
import subprocess
import platform

NVCC = os.environ.get('NVCC', 'nvcc')

if tuple.__itemsize__ == 4:
    OPT = '-m32 --compiler-options "-fPIC"'
elif tuple.__itemsize__ == 8:
    OPT = '-m64 --compiler-options "-fPIC"'

DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
CUB_INCLUDE = '-I%s/thirdparty/cub' % DIR
MGPU_INCLUDE = '-I%s/thirdparty/moderngpu/include' % DIR

GENCODE_SMXX = "-gencode arch=compute_{CC},code=sm_{CC}"
GENCODE_SM20 = GENCODE_SMXX.format(CC=20)
GENCODE_SM30 = GENCODE_SMXX.format(CC=30)
GENCODE_SM35 = GENCODE_SMXX.format(CC=35)
GENCODE_SM50 = GENCODE_SMXX.format(CC=50)
GENCODE_COMPUTEXX = "-gencode arch=compute_{CC},code=compute_{CC}"
GENCODE_COMPUTE50 = GENCODE_COMPUTEXX.format(CC=50)


SM = []
SM.append(GENCODE_SM20)
SM.append(GENCODE_SM30)
SM.append(GENCODE_SM35)
SM.append(GENCODE_SM50)
SM.append(GENCODE_COMPUTE50)

GENCODE_FLAGS = ' '.join(SM)

LIBDIR = '%s/cudalib/lib' % DIR

def run_shell(cmd):
    print(cmd)
    subprocess.check_call(cmd, shell=True)

def library_extension():
    p = platform.system()
    if p == 'Linux':
        return 'so'
    if p == 'Windows':
        return 'dll'
    if p == 'Darwin':
        return 'dylib'

def build_cuda(srcdir, out, ins, includes):
    ext = library_extension()
    output = os.path.join(LIBDIR, '%s.%s' % (out, ext))
    inputs = ' '.join([os.path.join(srcdir, p)
                       for p in ins])
    argtemp = '{opt} {inc} -O3 {gen} --shared -o {out} {inp}'
    args = argtemp.format(inc=includes, gen=GENCODE_FLAGS, out=output,
                          inp=inputs, opt=OPT)
    cmd = ' '.join([NVCC, args])
    run_shell(cmd)


def build_radixsort():
    build_cuda(srcdir='src',
               out='nbpro_radixsort',
               ins=['cubradixsort.cu'],
               includes=CUB_INCLUDE, )


def build_mgpusort():
    build_cuda(srcdir='src',
               out='nbpro_segsort',
               ins=['mgpusort.cu'],
               includes=MGPU_INCLUDE, )


def ensure_libdir():
    if os.path.exists(LIBDIR):
        if not os.path.isdir(LIBDIR):
            raise RuntimeError('Lib dir is a file')
    else:
        os.mkdir(LIBDIR)

if __name__ == '__main__':
    ensure_libdir()
    build_radixsort()
    build_mgpusort()
