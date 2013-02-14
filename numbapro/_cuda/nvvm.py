'''
This is a direct translation of nvvm.h
'''

import sys, os
from ctypes import *
from error import NvvmError, NvvmSupportError
from numbapro._utils import finalizer

ADDRSPACE_GENERIC = 0
ADDRSPACE_GLOBAL = 1
ADDRSPACE_SHARED = 3
ADDRSPACE_CONSTANT = 4
ADDRSPACE_LOCAL = 5

# Opaque handle for comilation unit
nvvm_cu = c_void_p

# Result code
nvvm_result = c_int


RESULT_CODE_NAMES = '''
NVVM_SUCCESS
NVVM_ERROR_OUT_OF_MEMORY
NVVM_ERROR_NOT_INITIALIZED
NVVM_ERROR_ALREADY_INITIALIZED
NVVM_ERROR_CU_CREATION_FAILURE
NVVM_ERROR_IR_VERSION_MISMATCH
NVVM_ERROR_INVALID_INPUT
NVVM_ERROR_INVALID_CU
NVVM_ERROR_INVALID_IR
NVVM_ERROR_INVALID_OPTION
NVVM_ERROR_NO_MODULE_IN_CU
NVVM_ERROR_COMPILATION
'''.split()

for i, k in enumerate(RESULT_CODE_NAMES):
    setattr(sys.modules[__name__], k, i)


def find_libnvvm():
    from os.path import dirname, isfile, join

    # search locations
    search_paths = [
        dirname(__file__), # always look in this directory first
        join(sys.prefix, 'DLLs' if sys.platform == 'win32' else 'lib'),
    ]
    # determine DLL name
    dllname = {'linux2': 'libnvvm.so',
               'darwin': 'libnvvm.dylib',
               'win32': 'nvvm.dll'}[sys.platform]

    for dir_path in search_paths:
        path = join(dir_path, dllname)
        if isfile(path):
            return path

    raise Exception("Could not find %r in directories %r" % (dllname,
                                                             search_paths))


class NVVM(object):
    '''Process-wide singleton.
    '''
    _PROTOTYPES = {
        # nvvmResult nvvmInit()
        'nvvmInit': (nvvm_result,),

        # nvvmResult nvvmFini()
        'nvvmFini': (nvvm_result,),

        # nvvmResult nvvmVersion(int *major, int *minor)
        'nvvmVersion': (nvvm_result, POINTER(c_int), POINTER(c_int)),

        # nvvmResult nvvmCreateCU(nvvmCU *cu)
        'nvvmCreateCU': (nvvm_result, POINTER(nvvm_cu)),

        # nvvmResult nvvmDestroyCU(nvvmCU *cu)
        'nvvmDestroyCU': (nvvm_result, POINTER(nvvm_cu)),

        # nvvmResult nvvmCUAddModule(nvvmCU cu, const char *buffer, size_t size)
        'nvvmCUAddModule': (nvvm_result, nvvm_cu, c_char_p, c_size_t),

        # nvvmResult nvvmCompileCU(nvvmCU cu, int numOptions,
        #                          const char **options)
        'nvvmCompileCU': (nvvm_result, nvvm_cu, c_int, POINTER(c_char_p)),

        # nvvmResult nvvmGetCompiledResultSize(nvvmCU cu,
        #                                      size_t *bufferSizeRet)
        'nvvmGetCompiledResultSize': (nvvm_result, nvvm_cu, POINTER(c_size_t)),

        # nvvmResult nvvmGetCompiledResult(nvvmCU cu, char *buffer)
        'nvvmGetCompiledResult': (nvvm_result, nvvm_cu, c_char_p),

        # nvvmResult nvvmGetCompilationLogSize(nvvmCU cu,
        #                                      size_t *bufferSizeRet)
        'nvvmGetCompilationLogSize': (nvvm_result, nvvm_cu, POINTER(c_size_t)),

        # nvvmResult nvvmGetCompilationLog(nvvmCU cu, char *buffer)
        'nvvmGetCompilationLog': (nvvm_result, nvvm_cu, c_char_p),
    }

    # Singleton reference
    __INSTANCE = None

    def __new__(cls, override_path=None):
        if not cls.__INSTANCE:
            inst = cls.__INSTANCE = object.__new__(cls)

            if override_path is None:
                # Try to discover libNVVM automatically
                # Environment variable always override if present
                # and override_path is not defined.
                path = os.getenv('NUMBAPRO_NVVM') or find_libnvvm()
            else:
                path = override_path

            # Load the driver
            try:
                inst.driver = CDLL(path)
                inst.path = path
            except OSError:
                raise NvvmSupportError(
                      "libNVVM cannot be found. "
                      "Try setting environment variable NUMBAPRO_NVVM "
                      "with the path of the libNVVM shared library.")

            for name, proto in inst._PROTOTYPES.items():
                func = getattr(inst.driver, name)
                func.restype = proto[0]
                func.argtypes = proto[1:]
                setattr(inst, name, func)

            # Initialize
            err = inst.nvvmInit()
            inst.check_error(err, 'Failed to initialize NVVM.')

        return cls.__INSTANCE

    def __del__(self):
        err = self.driver.nvvmFini()
        self.check_error(err, 'Failde to finalize NVVM.', exit=True)

    def get_version(self):
        major = c_int()
        minor = c_int()
        err = self.nvvmVersion(byref(major), byref(minor))
        self.check_error(err, 'Failed to get version.')
        return major.value, minor.value

    def check_error(self, error, msg, exit=False):
        if error:
            exc = NvvmError(msg, RESULT_CODE_NAMES[error])
            if exit:
                print exc
                sys.exit(1)
            else:
                raise exc

class CompilationUnit(finalizer.OwnerMixin):
    def __init__(self):
        self.driver = NVVM()
        self._handle = nvvm_cu()
        err = self.driver.nvvmCreateCU(byref(self._handle))
        self.driver.check_error(err, 'Failed to create CU')
        self._finalizer_track(self._handle)

    @classmethod
    def _finalize(cls, handle):
        driver = NVVM()
        err = driver.nvvmDestroyCU(byref(handle))
        driver.check_error(err, 'Failed to destroy CU', exit=True)

    def add_module(self, llvmir):
        '''
         Add a module level NVVM IR to a compilation unit.
         - The buffer should contain an NVVM module IR either in the bitcode
           representation (LLVM3.0) or in the text representation.
        '''
        err = self.driver.nvvmCUAddModule(self._handle, llvmir, len(llvmir))
        self.driver.check_error(err, 'Failed to add module')

    def compile(self, **options):
        '''Perform Compliation

        The valid compiler options are

        target=<value>
          <value>: ptx (default), verify
        debug
        opt=<level>
          <level>: 0, 3 (default)
        arch=<arch_value>
          <arch_value>: compute_20 (default), compute_30
        ftz=<value>
          <value>: 0 (default, preserve denormal values, when performing
                      single-precision floating-point operations)
                   1 (flush denormal values to zero, when performing
                      single-precision floating-point operations)
        prec_sqrt=<value>
          <value>: 0 (use a faster approximation for single-precision
                      floating-point square root)
                   1 (default, use IEEE round-to-nearest mode for
                      single-precision floating-point square root)
        prec_div=<value>
          <value>: 0 (use a faster approximation for single-precision
                      floating-point division and reciprocals)
                   1 (default, use IEEE round-to-nearest mode for
                      single-precision floating-point division and reciprocals)
        fma=<value>
          <value>: 0 (disable FMA contraction),
                   1 (default, enable FMA contraction),
        '''

        # stringify options
        opts = []

        if options.get('target'):
            opts.append('-%s=%s' % (target, options.pop('target')))

        if options.get('debug'):
            opts.append('-g')
            options.pop('debug')

        if options.get('opt'):
            opts.append('-opt=%d' % options.pop('opt'))

        if options.get('arch'):
            opts.append('-arch=%s' % options.pop('arch'))

        for k in ('ftz', 'prec_sqrt', 'prec_div', 'fma'):
            if k in options:
                v = bool(options.pop(k))
                opts.append('-%s=%d' % (k.replace('_', '-'), v))

        # compile
        c_opts = (c_char_p * len(opts))(*map(c_char_p, opts))
        err = self.driver.nvvmCompileCU(self._handle, len(opts), c_opts)
        self.driver.check_error(err, 'Failed to compile')

        # get result
        reslen = c_size_t()
        err = self.driver.nvvmGetCompiledResultSize(self._handle, byref(reslen))
        self.driver.check_error(err, 'Failed to get size of compiled result.')

        ptxbuf = (c_char * reslen.value)()
        err = self.driver.nvvmGetCompiledResult(self._handle, ptxbuf)
        self.driver.check_error(err, 'Failed to get compiled result.')

        # get log
        err = self.driver.nvvmGetCompilationLogSize(self._handle, byref(reslen))
        self.driver.check_error(err, 'Failed to get compilation log size.')

        if reslen > 1:
            logbuf = (c_char * reslen.value)()
            err = self.driver.nvvmGetCompilationLog(self._handle, logbuf)
            self.driver.check_error(err, 'Failed to get compilation log.')

            self.log = logbuf[:] # popluate log attribute
        self.log = ''

        return ptxbuf[:]

data_layout = {32 : 'e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64',
64: 'e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64'}

default_data_layout = data_layout[sizeof(c_void_p) * 8]


def llvm_to_ptx(llvmir, **opts):
    cu = CompilationUnit()
    cu.add_module(llvmir)
    return cu.compile(**opts)

def set_cuda_kernel(lfunc):
    from llvm.core import MetaData, MetaDataString, Constant, Type
    m = lfunc.module

    ops = lfunc, MetaDataString.get(m, "kernel"), Constant.int(Type.int(), 1)
    md = MetaData.get(m, ops)

    nmd = m.get_or_insert_named_metadata('nvvm.annotations')
    nmd.add(md)

def fix_data_layout(module):
    module.data_layout = default_data_layout

