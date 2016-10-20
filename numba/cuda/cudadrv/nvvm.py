"""
This is a direct translation of nvvm.h
"""
from __future__ import print_function, absolute_import, division
import sys, logging, re
from ctypes import (c_void_p, c_int, POINTER, c_char_p, c_size_t, byref,
                    c_char)
import threading

from llvmlite import binding as ll
from llvmlite import ir

from numba import config, cgutils
from .error import NvvmError, NvvmSupportError
from .libs import get_libdevice, open_libdevice, open_cudalib

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


def is_available():
    """
    Return if libNVVM is available
    """
    try:
        NVVM()
    except NvvmSupportError:
        return False
    else:
        return True


_nvvm_lock = threading.Lock()

class NVVM(object):
    '''Process-wide singleton.
    '''
    _PROTOTYPES = {

        # nvvmResult nvvmVersion(int *major, int *minor)
        'nvvmVersion': (nvvm_result, POINTER(c_int), POINTER(c_int)),

        # nvvmResult nvvmIRVersion ( int* majorIR, int* minorIR, int* majorDbg, int* minorDbg )
        'nvvmIRVersion': (nvvm_result, POINTER(c_int), POINTER(c_int),
                          POINTER(c_int), POINTER(c_int)),

        # nvvmResult nvvmCreateProgram(nvvmProgram *cu)
        'nvvmCreateProgram': (nvvm_result, POINTER(nvvm_program)),

        # nvvmResult nvvmDestroyProgram(nvvmProgram *cu)
        'nvvmDestroyProgram': (nvvm_result, POINTER(nvvm_program)),

        # nvvmResult nvvmAddModuleToProgram(nvvmProgram cu, const char *buffer,
        #                                   size_t size, const char *name)
        'nvvmAddModuleToProgram': (
            nvvm_result, nvvm_program, c_char_p, c_size_t, c_char_p),

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

        # nvvmVerifyProgram
        'nvvmVerifyProgram': (nvvm_result, nvvm_program, c_int, POINTER(c_char_p))
    }

    # Singleton reference
    __INSTANCE = None

    def __new__(cls):
        with _nvvm_lock:
            if cls.__INSTANCE is None:
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

                # Setup compilation lock
                inst._lock = threading.Lock()

        return cls.__INSTANCE

    def get_version(self):
        major = c_int()
        minor = c_int()
        err = self.nvvmVersion(byref(major), byref(minor))
        self.check_error(err, 'Failed to get version.')
        return major.value, minor.value

    def get_ir_version(self):
        major = c_int()
        minor = c_int()
        majordbg = c_int()
        minordbg = c_int()
        err = self.nvvmIRVersion(byref(major), byref(minor),
                                 byref(majordbg), byref(minordbg))
        self.check_error(err, 'Failed to get IR version.')
        return (major.value, minor.value), (majordbg.value, minordbg.value)

    def check_error(self, error, msg, exit=False):
        if error:
            exc = NvvmError(msg, RESULT_CODE_NAMES[error])
            if exit:
                print(exc)
                sys.exit(1)
            else:
                raise exc

    def lock(self):
        return self._lock.acquire()

    def unlock(self):
        return self._lock.release()


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

    def __enter__(self):
        self.driver.lock()
        return self

    def __exit__(self, *args):
        self.driver.unlock()

    def add_module(self, buffer):
        """
         Add a module level NVVM IR to a compilation unit.
         - The buffer should contain an NVVM module IR either in the bitcode
           representation (LLVM3.0) or in the text representation.
        """
        err = self.driver.nvvmAddModuleToProgram(self._handle, buffer,
                                                 len(buffer), None)
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

        err = self.driver.nvvmVerifyProgram(self._handle, len(opts), c_opts)
        self._try_error(err, 'Failed to verify\n')

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

            return logbuf.value.decode('utf8')  # populate log attribute

        return ''


data_layout = {
    32: ('e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-'
         'f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64'),
    64: ('e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-'
         'f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64')}

default_data_layout = data_layout[tuple.__itemsize__ * 8]


# List of supported compute capability in sorted order
SUPPORTED_CC = (2, 0), (2, 1), (3, 0), (3, 5), (5, 0), (5, 2)


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
    return SUPPORTED_CC[-1]  # Choose the highest


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
files in the installation of CUDA.  (requires CUDA >=7.5)
'''

MISSING_LIBDEVICE_FILE_MSG = '''Missing libdevice file for {arch}.
Please ensure you have package cudatoolkit 7.5.
Install package by:

    conda install cudatoolkit=7.5
'''


class LibDevice(object):
    _cache_ = {}
    _known_arch = [
        "compute_20",
        "compute_30",
        "compute_35",
        "compute_50",
    ]

    def __init__(self, arch):
        """
        arch --- must be result from get_arch_option()
        """
        if arch not in self._cache_:
            arch = self._get_closest_arch(arch)
            if get_libdevice(arch) is None:
                raise RuntimeError(MISSING_LIBDEVICE_FILE_MSG.format(arch=arch))
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


# Translation of code from CUDA Programming Guide v6.5, section B.12

def make_ir_numba_atomic_double_add(m):
    double = ir.DoubleType()
    fnty = ir.FunctionType(double, (double.as_pointer(), double))
    fn = ir.Function(m, fnty, name='___numba_atomic_double_add')
    fn.attributes.add('alwaysinline')
    fn.linkage = 'linkonce_odr'

    label_entry = fn.append_basic_block('entry')
    label_attempt = fn.append_basic_block('attempt')
    label_done = fn.append_basic_block('done')

    builder = ir.IRBuilder(label_entry)
    [ptr, val] = fn.args
    ptr.name = 'ptr'
    val.name = 'val'

    i64 = ir.IntType(64)
    i64ptr = i64.as_pointer()
    iptr = builder.bitcast(ptr, i64ptr)
    old2 = builder.load(iptr)
    old2.volatile = True

    builder.branch(label_attempt)

    builder.position_at_end(label_attempt)
    old = builder.phi(i64)
    old.add_incoming(old2, label_entry)
    dold = builder.bitcast(old, double)
    dnew = builder.fadd(dold, val)
    new = builder.bitcast(dnew, i64)
    cmpxchg = builder.cmpxchg(iptr, old, new, ordering='monotonic')
    cmpxchg.volatile = True
    cas, ok = cgutils.unpack_tuple(builder, cmpxchg)
    old.add_incoming(cas, label_attempt)
    repeat = builder.not_(ok)
    builder.cbranch(repeat, label_attempt, label_done)

    builder.position_at_end(label_done)
    result = builder.bitcast(old, double)
    builder.ret(result)


def make_ir_numba_atomic_double_max(m):
    double = ir.DoubleType()
    i64 = ir.IntType(64)
    i64ptr = i64.as_pointer()
    fnty = ir.FunctionType(double, [double.as_pointer(), double])
    fn = ir.Function(m, fnty, name='___numba_atomic_double_max')
    fn.attributes.add('alwaysinline')
    fn.linakge = 'internal'

    label_entry = fn.append_basic_block('entry')
    label_lt_check = fn.append_basic_block('lt_check')
    label_attempt = fn.append_basic_block('attempt')
    label_done = fn.append_basic_block('done')

    [ptr, val] = fn.args
    ptr.name = 'ptr'
    val.name = 'val'

    builder = ir.IRBuilder(label_entry)
    ptrval = builder.load(ptr)
    ptrval.volatile = True
    # Check if val is a NaN and return *ptr early if so
    valnan = builder.fcmp_unordered('uno', val, val)
    builder.cbranch(valnan, label_done, label_lt_check)

    builder.position_at_end(label_lt_check)
    dold = builder.phi(double)
    dold.add_incoming(ptrval, label_entry)
    # Continue attempts if dold < val or dold is NaN (using ult semantics)
    lt = builder.fcmp_unordered('<', dold, val)
    builder.cbranch(lt, label_attempt, label_done)

    builder.position_at_end(label_attempt)
    # Attempt to swap in the larger value
    iold = builder.bitcast(dold, i64)
    iptr = builder.bitcast(ptr, i64ptr)
    ival = builder.bitcast(val, i64)
    cmpxchg = builder.cmpxchg(iptr, iold, ival, ordering='monotonic')
    cmpxchg.volatile = True
    cas, ok = cgutils.unpack_tuple(builder, cmpxchg)
    dcas = builder.bitcast(cas, double)
    dold.add_incoming(dcas, label_attempt)
    builder.branch(label_lt_check)

    builder.position_at_end(label_done)
    ret = builder.phi(double)
    ret.add_incoming(ptrval, label_entry)
    ret.add_incoming(dold, label_lt_check)
    builder.ret(ret)


def make_helper_lib():
    m = ir.Module()
    make_ir_numba_atomic_double_add(m)
    make_ir_numba_atomic_double_max(m)
    return ll.parse_assembly(str(m))


def _replace_datalayout(llvmir):
    """
    Find the line containing the datalayout and replace it
    """
    lines = llvmir.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("target datalayout"):
            tmp = 'target datalayout = "{0}"'
            lines[i] = tmp.format(default_data_layout)
            break
    return '\n'.join(lines)

_helper_lib_module = make_helper_lib()

_compiler_lock = threading.Lock()


def llvm_to_ptx(llvmir, **opts):
    with CompilationUnit() as cu:
        libdevice = LibDevice(arch=opts.get('arch', 'compute_20'))
        # New LLVM generate a shorthand for datalayout that NVVM does not know
        llvmir = _replace_datalayout(llvmir)

        llmod = ll.parse_assembly(llvmir)
        llmod.link_in(_helper_lib_module, preserve=True)
        llvmbitcode = llmod.as_bitcode()

        cu.add_module(llvmbitcode)
        cu.add_module(libdevice.get())
        ptx = cu.compile(**opts)
        return ptx


re_metadata_def = re.compile(r"\!\d+\s*=")
re_metadata_correct_usage = re.compile(r"metadata\s*\![{'\"]")

re_attributes_def = re.compile(r"^attributes #\d+ = \{ ([\w\s]+)\ }")
unsupported_attributes = {'argmemonly', 'inaccessiblememonly',
                          'inaccessiblemem_or_argmemonly', 'norecurse'}

re_getelementptr = re.compile(r"\Wgetelementptr\s(?:inbounds )?\(?")

re_load = re.compile(r"\Wload\s(?:volatile )?")

re_call = re.compile(r"(call\s[^@]+\))(\s@)")

re_type_tok = re.compile(r"[,{}()[\]]")

re_annotations = re.compile(r"\Wnonnull\W")


def llvm38_to_34_ir(ir):
    """
    Convert LLVM 3.8 IR for LLVM 3.4.
    """
    def parse_out_leading_type(s):
        par_level = 0
        pos = 0
        # Parse out the first <ty> (which may be an aggregate type)
        while True:
            m = re_type_tok.search(s, pos)
            if m is None:
                # End of line
                raise RuntimeError("failed parsing leading type: %s" % (s,))
                break
            pos = m.end()
            tok = m.group(0)
            if tok == ',':
                if par_level == 0:
                    # End of operand
                    break
            elif tok in '{[(':
                par_level += 1
            elif tok in ')]}':
                par_level -= 1
        return s[pos:].lstrip()

    buf = []
    for line in ir.splitlines():
        orig = line
        if re_metadata_def.match(line):
            # Rewrite metadata since LLVM 3.7 dropped the "metadata" type prefix.
            if None is re_metadata_correct_usage.search(line):
                line = line.replace('!{', 'metadata !{')
                line = line.replace('!"', 'metadata !"')
        if line.startswith('attributes #'):
            # Remove function attributes unsupported pre-3.8
            m = re_attributes_def.match(line)
            attrs = m.group(1).split()
            attrs = ' '.join(a for a in attrs if a not in unsupported_attributes)
            line = line.replace(m.group(1), attrs)
        if 'getelementptr ' in line:
            # Rewrite "getelementptr ty, ty* ptr, ..."
            # to "getelementptr ty *ptr, ..."
            m = re_getelementptr.search(line)
            if m is None:
                raise RuntimeError("failed parsing getelementptr: %s" % (line,))
            pos = m.end()
            line = line[:pos] + parse_out_leading_type(line[pos:])
        if 'load ' in line:
            # Rewrite "load ty, ty* ptr"
            # to "load ty *ptr"
            m = re_load.search(line)
            if m is None:
                raise RuntimeError("failed parsing load: %s" % (line,))
            pos = m.end()
            line = line[:pos] + parse_out_leading_type(line[pos:])
        if 'call ' in line:
            # Rewrite "call ty (...) @foo"
            # to "call ty (...)* @foo"
            line = re_call.sub(r"\1*\2", line)

        # Remove unknown annotations
        line = re_annotations.sub('', line)

        buf.append(line)

    return '\n'.join(buf)


def set_cuda_kernel(lfunc):
    from llvmlite.llvmpy.core import MetaData, MetaDataString, Constant, Type

    m = lfunc.module

    ops = lfunc, MetaDataString.get(m, "kernel"), Constant.int(Type.int(), 1)
    md = MetaData.get(m, ops)

    nmd = m.get_or_insert_named_metadata('nvvm.annotations')
    nmd.add(md)

    # set nvvm ir version
    i32 = ir.IntType(32)
    md_ver = m.add_metadata([i32(1), i32(3)])
    m.add_named_metadata('nvvmir.version', md_ver)


def fix_data_layout(module):
    module.data_layout = default_data_layout
