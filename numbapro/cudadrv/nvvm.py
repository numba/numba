'''
This is a direct translation of nvvm.h
'''

import sys, os
from ctypes import (c_void_p, c_int, POINTER, c_char_p, c_size_t, CDLL, byref,
                    c_char)
from error import NvvmError, NvvmSupportError
from numbapro._utils import finalizer

ADDRSPACE_GENERIC = 0
ADDRSPACE_GLOBAL = 1
ADDRSPACE_SHARED = 3
ADDRSPACE_CONSTANT = 4
ADDRSPACE_LOCAL = 5

# Opaque handle for comilation unit
nvvm_program = c_void_p

# Result code
nvvm_result = c_int


RESULT_CODE_NAMES = '''
NVVM_SUCCESS
NVVM_ERROR_OUT_OF_MEMORY
NVVM_ERROR_PROGRAM_CREATION_FAILURE
NVVM_ERROR_IR_VERSION_MISMATCH
NVVM_ERROR_INVALID_INPUT
NVVM_ERROR_INVALID_PROGRAM
NVVM_ERROR_INVALID_IR
NVVM_ERROR_INVALID_OPTION
NVVM_ERROR_NO_MODULE_IN_PROGRAM
NVVM_ERROR_COMPILATION
'''.split()

for i, k in enumerate(RESULT_CODE_NAMES):
    setattr(sys.modules[__name__], k, i)

class NVVM(object):
    '''Process-wide singleton.
    '''
    _PROTOTYPES = {

        # nvvmResult nvvmVersion(int *major, int *minor)
        'nvvmVersion': (nvvm_result, POINTER(c_int), POINTER(c_int)),

        # nvvmResult nvvmCreateProgram(nvvmProgram *cu)
        'nvvmCreateProgram': (nvvm_result, POINTER(nvvm_program)),

        # nvvmResult nvvmDestroyProgram(nvvmProgram *cu)
        'nvvmDestroyProgram': (nvvm_result, POINTER(nvvm_program)),

        # nvvmResult nvvmAddModuleToProgram(nvvmProgram cu, const char *buffer, size_t size)
        'nvvmAddModuleToProgram': (nvvm_result, nvvm_program, c_char_p, c_size_t),

        # nvvmResult nvvmCompileProgram(nvvmProgram cu, int numOptions,
        #                          const char **options)
        'nvvmCompileProgram': (nvvm_result, nvvm_program, c_int, POINTER(c_char_p)),

        # nvvmResult nvvmGetCompiledResultSize(nvvmProgram cu,
        #                                      size_t *bufferSizeRet)
        'nvvmGetCompiledResultSize': (nvvm_result, nvvm_program, POINTER(c_size_t)),

        # nvvmResult nvvmGetCompiledResult(nvvmProgram cu, char *buffer)
        'nvvmGetCompiledResult': (nvvm_result, nvvm_program, c_char_p),

        # nvvmResult nvvmGetProgramLogSize(nvvmProgram cu,
        #                                      size_t *bufferSizeRet)
        'nvvmGetProgramLogSize': (nvvm_result, nvvm_program, POINTER(c_size_t)),

        # nvvmResult nvvmGetProgramLog(nvvmProgram cu, char *buffer)
        'nvvmGetProgramLog': (nvvm_result, nvvm_program, c_char_p),
    }

    # Singleton reference
    __INSTANCE = None

    def __new__(cls, override_path=None):
        if not cls.__INSTANCE:
            from numbapro import findlib
            # Load the driver
            where = ([override_path]
                        if override_path is not None
                        else findlib.find_lib('nvvm', 'NUMBAPRO_NVVM'))
            for path in where:
                try:
                    driver = CDLL(path)
                    inst = cls.__INSTANCE = object.__new__(cls)
                    inst.driver = driver
                    inst.path = path
                except OSError:
                    pass # continue
                else:
                    break # got it, break out
            else:
                cls.__INSTANCE = None
                raise NvvmSupportError(
                           "libNVVM cannot be found. "
                           "Try setting environment variable NUMBAPRO_NVVM "
                           "with the path of the libNVVM shared library.")

            # Find & populate functions
            for name, proto in inst._PROTOTYPES.items():
                func = getattr(inst.driver, name)
                func.restype = proto[0]
                func.argtypes = proto[1:]
                setattr(inst, name, func)

        return cls.__INSTANCE

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
        self._handle = nvvm_program()
        err = self.driver.nvvmCreateProgram(byref(self._handle))
        self.driver.check_error(err, 'Failed to create CU')
        self._finalizer_track(self._handle)

    @classmethod
    def _finalize(cls, handle):
        driver = NVVM()
        err = driver.nvvmDestroyProgram(byref(handle))
        driver.check_error(err, 'Failed to destroy CU', exit=True)

    def add_module(self, llvmir):
        '''
         Add a module level NVVM IR to a compilation unit.
         - The buffer should contain an NVVM module IR either in the bitcode
           representation (LLVM3.0) or in the text representation.
        '''
        err = self.driver.nvvmAddModuleToProgram(self._handle, llvmir, len(llvmir))
        self.driver.check_error(err, 'Failed to add module')

    def compile(self, **options):
        '''Perform Compliation

        The valid compiler options are

         *   - -g (enable generation of debugging information)
         *   - -opt=
         *     - 0 (disable optimizations)
         *     - 3 (default, enable optimizations)
         *   - -arch=
         *     - compute_20 (default)
         *     - compute_30
         *     - compute_35
         *   - -ftz=
         *     - 0 (default, preserve denormal values, when performing
         *          single-precision floating-point operations)
         *     - 1 (flush denormal values to zero, when performing
         *          single-precision floating-point operations)
         *   - -prec-sqrt=
         *     - 0 (use a faster approximation for single-precision
         *          floating-point square root)
         *     - 1 (default, use IEEE round-to-nearest mode for
         *          single-precision floating-point square root)
         *   - -prec-div=
         *     - 0 (use a faster approximation for single-precision
         *          floating-point division and reciprocals)
         *     - 1 (default, use IEEE round-to-nearest mode for
         *          single-precision floating-point division and reciprocals)
         *   - -fma=
         *     - 0 (disable FMA contraction)
         *     - 1 (default, enable FMA contraction)
         *        
         '''

        # stringify options
        opts = []

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
        err = self.driver.nvvmCompileProgram(self._handle, len(opts), c_opts)
        self.driver.check_error(err, 'Failed to compile')

        # get result
        reslen = c_size_t()
        err = self.driver.nvvmGetCompiledResultSize(self._handle, byref(reslen))
        self.driver.check_error(err, 'Failed to get size of compiled result.')

        ptxbuf = (c_char * reslen.value)()
        err = self.driver.nvvmGetCompiledResult(self._handle, ptxbuf)
        self.driver.check_error(err, 'Failed to get compiled result.')

        # get log
        err = self.driver.nvvmGetProgramLogSize(self._handle, byref(reslen))
        self.driver.check_error(err, 'Failed to get compilation log size.')

        if reslen > 1:
            logbuf = (c_char * reslen.value)()
            err = self.driver.nvvmGetProgramLog(self._handle, logbuf)
            self.driver.check_error(err, 'Failed to get compilation log.')

            self.log = logbuf[:] # popluate log attribute
        self.log = ''

        return ptxbuf[:]

data_layout = {
32: ('e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-'
    'f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64'),
64: ('e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-'
     'f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64')}

default_data_layout = data_layout[tuple.__itemsize__ * 8]

SUPPORTED_CC = frozenset([(2, 0), (3, 0), (3, 5)])

def get_arch_option(major, minor):
    if major == 2:
        minor = 0
    if major == 3:
        if minor < 5:
            minor = 0
        else:
            minor = 5
    if (major, minor) not in SUPPORTED_CC:
        raise Exception("compute compability %d.%d is not supported" %
                        (major, minor))
    arch = 'compute_%d%d' % (major, minor)
    return arch

MISSING_LIBDEVICE_MSG = '''
Please define environment variable NUMBAPRO_LIBDEVICE=/path/to/libdevice 
/path/to/libdevice -- is the path to the directory containing the libdevice.*.bc
files in the installation of CUDA.  (requires CUDA >=5.5)
'''

class LibDevice(object):
    _cache_ = {}
    def __init__(self, arch):
        '''
        arch --- must be result from get_arch_option()
        '''
        if arch not in self._cache_:
            try:
                libdevice_dir = os.environ['NUMBAPRO_LIBDEVICE']
            except KeyError:
                from numbapro import findlib
                libdevice_dir = findlib.get_lib_dir()

            prefix_template = 'libdevice.%s'
            ext = '.bc'

            candidates = []
            for fname in os.listdir(libdevice_dir):
                prefix = prefix_template % arch
                if fname.startswith(prefix) and fname.endswith(ext):
                    chosen = os.path.join(libdevice_dir, fname)
                    candidates.append(chosen)

            if not candidates:
                raise Exception(MISSING_LIBDEVICE_MSG)

            chosen = max(candidates)
            with open(chosen, 'rb') as bcfile:
                bc = bcfile.read()
                self._cache_[arch] = bc

        self.bc = self._cache_[arch]

    def get(self):
        return self.bc

def llvm_to_ptx(llvmir, **opts):
    cu = CompilationUnit()
    libdevice = LibDevice(arch=opts.get('arch', 'compute_20'))
    cu.add_module(llvmir)
    cu.add_module(libdevice.get())
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

