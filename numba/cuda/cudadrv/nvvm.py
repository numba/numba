"""
This is a direct translation of nvvm.h
"""
from __future__ import print_function, absolute_import, division
import sys, logging, re
from ctypes import (c_void_p, c_int, POINTER, c_char_p, c_size_t, byref,
                    c_char)
from numba import config
from .error import NvvmError, NvvmSupportError
from .libs import open_libdevice, open_cudalib

logger = logging.getLogger(__name__)

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
        'nvvmAddModuleToProgram': (
            nvvm_result, nvvm_program, c_char_p, c_size_t),

        # nvvmResult nvvmCompileProgram(nvvmProgram cu, int numOptions,
        #                          const char **options)
        'nvvmCompileProgram': (
            nvvm_result, nvvm_program, c_int, POINTER(c_char_p)),

        # nvvmResult nvvmGetCompiledResultSize(nvvmProgram cu,
        #                                      size_t *bufferSizeRet)
        'nvvmGetCompiledResultSize': (
            nvvm_result, nvvm_program, POINTER(c_size_t)),

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

    def __new__(cls):
        if not cls.__INSTANCE:
            cls.__INSTANCE = inst = object.__new__(cls)
            try:
                inst.driver = open_cudalib('nvvm', ccc=True)
            except OSError as e:
                cls.__INSTANCE = None
                errmsg = ("libNVVM cannot be found. Do `conda install "
                          "cudatoolkit`:\n%s")
                raise NvvmSupportError(errmsg % e)

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
                print(exc)
                sys.exit(1)
            else:
                raise exc


class CompilationUnit(object):
    def __init__(self):
        self.driver = NVVM()
        self._handle = nvvm_program()
        err = self.driver.nvvmCreateProgram(byref(self._handle))
        self.driver.check_error(err, 'Failed to create CU')

    def __del__(self):
        driver = NVVM()
        err = driver.nvvmDestroyProgram(byref(self._handle))
        driver.check_error(err, 'Failed to destroy CU', exit=True)

    def add_module(self, buffer):
        """
         Add a module level NVVM IR to a compilation unit.
         - The buffer should contain an NVVM module IR either in the bitcode
           representation (LLVM3.0) or in the text representation.
        """
        err = self.driver.nvvmAddModuleToProgram(self._handle, buffer,
                                                 len(buffer))
        self.driver.check_error(err, 'Failed to add module')

    def compile(self, **options):
        """Perform Compliation

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
         """

        # stringify options
        opts = []

        if options.get('debug'):
            opts.append('-g')
            options.pop('debug')

        if options.get('opt'):
            opts.append('-opt=%d' % options.pop('opt'))

        if options.get('arch'):
            opts.append('-arch=%s' % options.pop('arch'))

        other_options = (
            'ftz',
            'prec_sqrt',
            'prec_div',
            'fma',
        )

        for k in other_options:
            if k in options:
                v = int(bool(options.pop(k)))
                opts.append('-%s=%d' % (k.replace('_', '-'), v))

        # If there are any option left
        if options:
            optstr = ', '.join(map(repr, options.keys()))
            raise NvvmError("unsupported option {0}".format(optstr))

        # compile
        c_opts = (c_char_p * len(opts))(*[c_char_p(x.encode('utf8'))
                                          for x in opts])
        err = self.driver.nvvmCompileProgram(self._handle, len(opts), c_opts)
        self._try_error(err, 'Failed to compile\n')

        # get result
        reslen = c_size_t()
        err = self.driver.nvvmGetCompiledResultSize(self._handle, byref(reslen))

        self._try_error(err, 'Failed to get size of compiled result.')

        ptxbuf = (c_char * reslen.value)()
        err = self.driver.nvvmGetCompiledResult(self._handle, ptxbuf)
        self._try_error(err, 'Failed to get compiled result.')

        # get log
        self.log = self.get_log()

        return ptxbuf[:]

    def _try_error(self, err, msg):
        self.driver.check_error(err, "%s\n%s" % (msg, self.get_log()))

    def get_log(self):
        reslen = c_size_t()
        err = self.driver.nvvmGetProgramLogSize(self._handle, byref(reslen))
        self.driver.check_error(err, 'Failed to get compilation log size.')

        if reslen.value > 1:
            logbuf = (c_char * reslen.value)()
            err = self.driver.nvvmGetProgramLog(self._handle, logbuf)
            self.driver.check_error(err, 'Failed to get compilation log.')

            return logbuf.value.decode('utf8') # popluate log attribute

        return ''


data_layout = {
    32: ('e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-'
         'f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64'),
    64: ('e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-'
         'f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64')}

default_data_layout = data_layout[tuple.__itemsize__ * 8]


# List of supported compute capability in sorted order
SUPPORTED_CC = (2, 0), (2, 1), (3, 0), (3, 5), (5, 0)


def _find_arch(mycc):
    for i, cc in enumerate(SUPPORTED_CC):
        if cc == mycc:
            # Matches
            return cc
        elif cc > mycc:
            # Exceeded
            if i == 0:
                # CC lower than supported
                raise NvvmSupportError("GPU compute capability %d.%d is "
                                       "not supported (requires >=2.0)" % mycc)
            else:
                # return the previous CC
                return SUPPORTED_CC[i - 1]

    # CC higher than supported
    return SUPPORTED_CC[-1]   # Choose the highest


def get_arch_option(major, minor):
    """Matches with the closest architecture option
    """
    if config.FORCE_CUDA_CC:
        arch = config.FORCE_CUDA_CC
    else:
        arch = _find_arch((major, minor))
    return 'compute_%d%d' % arch


MISSING_LIBDEVICE_MSG = '''
Please define environment variable NUMBAPRO_LIBDEVICE=/path/to/libdevice
/path/to/libdevice -- is the path to the directory containing the libdevice.*.bc
files in the installation of CUDA.  (requires CUDA >=5.5)
'''


class LibDevice(object):
    _cache_ = {}
    _known_arch = [
        "compute_20",
        "compute_30",
        "compute_35",
    ]

    def __init__(self, arch):
        """
        arch --- must be result from get_arch_option()
        """
        if arch not in self._cache_:
            arch = self._get_closest_arch(arch)
            self._cache_[arch] = open_libdevice(arch)

        self.arch = arch
        self.bc = self._cache_[arch]

    def _get_closest_arch(self, arch):
        res = self._known_arch[0]
        for potential in self._known_arch:
            if arch >= potential:
                res = potential
        return res

    def get(self):
        return self.bc


ir_numba_cas_hack = """
define internal i32 @___numba_cas_hack(i32* %ptr, i32 %cmp, i32 %val) alwaysinline {
    %out = cmpxchg volatile i32* %ptr, i32 %cmp, i32 %val monotonic
    ret i32 %out
}
"""

# Translation of code from CUDA Programming Guide v6.5, section B.12
ir_numba_atomic_double_add = """
define internal double @___numba_atomic_double_add(double* %ptr, double %val) alwaysinline {
entry:
    %iptr = bitcast double* %ptr to i64*
    %old2 = load volatile i64* %iptr
    br label %attempt

attempt:
    %old = phi i64 [ %old2, %entry ], [ %cas, %attempt ]
    %dold = bitcast i64 %old to double
    %dnew = fadd double %dold, %val
    %new = bitcast double %dnew to i64
    %cas = cmpxchg volatile i64* %iptr, i64 %old, i64 %new monotonic
    %repeat = icmp ne i64 %cas, %old
    br i1 %repeat, label %attempt, label %done

done:
    %result = bitcast i64 %old to double
    ret double %result
}
"""

def llvm_to_ptx(llvmir, **opts):
    cu = CompilationUnit()
    libdevice = LibDevice(arch=opts.get('arch', 'compute_20'))
    # New LLVM generate a shorthand for datalayout that NVVM does not know
    llvmir = llvmir.replace('e-i64:64-v16:16-v32:32-n16:32:64',
                            default_data_layout)

    # Replace with our cmpxchg and atomic implementations because LLVM 3.5 has
    # a new semantic for cmpxchg.
    replacements = [
        ('declare i32 @___numba_cas_hack(i32*, i32, i32)',
            ir_numba_cas_hack),
        ('declare double @___numba_atomic_double_add(double*, double)',
            ir_numba_atomic_double_add)]

    for decl, fn in replacements:
        llvmir = llvmir.replace(decl, fn)

    llvmir = llvm33_to_32_ir(llvmir)
    cu.add_module(llvmir.encode('utf8'))
    cu.add_module(libdevice.get())
    ptx = cu.compile(**opts)
    return ptx


re_fnattr_ref = re.compile('#\d+')
re_fnattr_def = re.compile('attributes\s+(#\d+)\s*=\s*{((?:\s*\w+)+)\s*}')


def llvm33_to_32_ir(ir):
    """rewrite function attributes in the IR
    """

    invalid_attrs = frozenset(['noduplicate'])

    attrs = {}
    for m in re_fnattr_def.finditer(ir):
        ct, text = m.groups()
        attrs[ct] = ' '.join(set(text.split()) - invalid_attrs)

    def scanline(line):
        if line.startswith('define') or line.startswith('declare'):
            for k, v in attrs.items():
                if k in line:
                    return line.replace(k, v)
        elif re_fnattr_def.match(line):
            return '; %s' % line
        return line

    return '\n'.join(scanline(ln) for ln in ir.splitlines())


def set_cuda_kernel(lfunc):
    from llvmlite.llvmpy.core import MetaData, MetaDataString, Constant, Type

    m = lfunc.module

    ops = lfunc, MetaDataString.get(m, "kernel"), Constant.int(Type.int(), 1)
    md = MetaData.get(m, ops)

    nmd = m.get_or_insert_named_metadata('nvvm.annotations')
    nmd.add(md)


def fix_data_layout(module):
    module.data_layout = default_data_layout

