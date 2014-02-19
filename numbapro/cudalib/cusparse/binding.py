from __future__ import print_function, absolute_import, division

import numpy as np
from ctypes import c_float, c_double, byref, c_int, Structure, c_void_p, POINTER

from numbapro.cudalib.libutils import Lib, ctype_function
from numbapro.cudadrv.driver import cu_stream, device_pointer, host_pointer
from numbapro._utils import finalizer
from numbapro.cudalib.cctypes import c_complex, c_double_complex

from . import decls

INV_STATUS = dict(
    CUSPARSE_STATUS_SUCCESS=0,
    CUSPARSE_STATUS_NOT_INITIALIZED=1,
    CUSPARSE_STATUS_ALLOC_FAILED=2,
    CUSPARSE_STATUS_INVALID_VALUE=3,
    CUSPARSE_STATUS_ARCH_MISMATCH=4,
    CUSPARSE_STATUS_MAPPING_ERROR=5,
    CUSPARSE_STATUS_EXECUTION_FAILED=6,
    CUSPARSE_STATUS_INTERNAL_ERROR=7,
    CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED=8
)

STATUS = dict((v, k) for k, v in INV_STATUS.items())

CUSPARSE_INDEX_BASE_ZERO = 0
CUSPARSE_INDEX_BASE_ONE = 1

CUSPARSE_DIAG_TYPE_NON_UNIT = 0
CUSPARSE_DIAG_TYPE_UNIT = 1

CUSPARSE_FILL_MODE_LOWER = 0
CUSPARSE_FILL_MODE_UPPER = 1

CUSPARSE_MATRIX_TYPE_GENERAL = 0
CUSPARSE_MATRIX_TYPE_SYMMETRIC = 1
CUSPARSE_MATRIX_TYPE_HERMITIAN = 2
CUSPARSE_MATRIX_TYPE_TRIANGULAR = 3

CUSPARSE_ACTION_SYMBOLIC = 0
CUSPARSE_ACTION_NUMERIC = 1

CUSPARSE_POINTER_MODE_HOST = 0
CUSPARSE_POINTER_MODE_DEVICE = 1

CUSPARSE_OPERATION_NON_TRANSPOSE = 0
CUSPARSE_OPERATION_TRANSPOSE = 1
CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2

CUSPARSE_DIRECTION_ROW = 0
CUSPARSE_DIRECTION_COLUMN = 1

# automatically decide how to split the
# data into regular/irregular part
CUSPARSE_HYB_PARTITION_AUTO = 0
# store data into regular part up to a user
# specified treshhold
CUSPARSE_HYB_PARTITION_USER = 1
# store all data in the regular part
CUSPARSE_HYB_PARTITION_MAX = 2

cusparseHandle_t = c_void_p
cusparseMatDescr_t = c_void_p
cusparseSolveAnalysisInfo_t = c_void_p
cusparseHybMat_t = c_void_p

cusparseStatus_t = c_int
cusparseIndexBase_t = c_int
cusparsePointerMode_t = c_int
cusparseAction_t = c_int
cusparseFillMode_t = c_int
cusparseDiagType_t = c_int
cusparseOperation_t = c_int
cusparseDirection_t = c_int
cusparseHybPartition_t = c_int
cusparseMatrixType_t = c_int


_c_types = {
    'int'                           : c_int,
    'float'                         : c_float,
    'double'                        : c_double,
    'cuComplex'                     : c_complex,
    'cuDoubleComplex'               : c_double_complex,
    'cudaStream_t'                  : cu_stream,
    'cusparseStatus_t'              : cusparseStatus_t,
    'cusparseMatDescr_t'            : cusparseMatDescr_t,
    'cusparseSolveAnalysisInfo_t'   : cusparseSolveAnalysisInfo_t,
    'cusparseHybMat_t'              : cusparseHybMat_t,
    'cusparseHandle_t'              : cusparseHandle_t,
    'cusparsePointerMode_t'         : cusparsePointerMode_t,
    'cusparseAction_t'              : cusparseAction_t,
    'cusparseFillMode_t'            : cusparseFillMode_t,
    'cusparseDiagType_t'            : cusparseDiagType_t,
    'cusparseOperation_t'           : cusparseOperation_t,
    'cusparseDirection_t'           : cusparseDirection_t,
    'cusparseHybPartition_t'        : cusparseHybPartition_t,
    'cusparseIndexBase_t'           : cusparseIndexBase_t,
    'cusparseMatrixType_t'          : cusparseMatrixType_t,
}


class CuSparseError(Exception):
    def __init__(self, code):
        super(CuSparseError, self).__init__(STATUS[code])


def _get_type(k):
    try:
        return _c_types[k]
    except KeyError:
        if k[-1] == '*':
            return POINTER(_get_type(k[:-1]))
        raise


def _init_ctype_function(name, decl):
    res, args = decl
    types = [_get_type(a) for _, a in args]
    return ctype_function(_get_type(res), *types)


def _declarations():
    for k in dir(decls):
        if not k.startswith('_'):
            yield k, getattr(decls, k)


class _libcusparse(Lib):
    lib = 'cusparse'
    ErrorType = CuSparseError


def _init_libcusparse():
    gv = {}
    for k, v in _declarations():
        gv[k] = _init_ctype_function(k, v)
    base = _libcusparse
    return type('libcusparse', (base,), gv)

libcusparse = _init_libcusparse()


class _cuSparse(finalizer.OwnerMixin):
    def __init__(self):
        self._api = libcusparse()
        self._handle = cusparseHandle_t()
        self._api.cusparseCreate(byref(self._handle))
        self._finalizer_track((self._handle, self._api))
        self._stream = 0

    @classmethod
    def _finalize(cls, res):
        handle, api = res
        api.cusparseDestroy(handle)

    @property
    def version(self):
        ver = c_int()
        self._api.cusparseGetVersion(self._handle, byref(ver))
        return ver.value

    @property
    def stream(self):
        return self._stream

    @stream.setter
    def stream(self, stream):
        self._stream = stream
        self._api.cusparseSetStream(self._handle, self._stream._handle)


_strip_prefix = 'cusparse'
_len_strip_prefix = len(_strip_prefix)


def mangle(name):
    assert name.startswith(_strip_prefix)
    name = name[_len_strip_prefix:]
    return name


def _flatten_args(args, kws, argnames):
    values = list(args)
    for name in argnames[len(args):]:
        if name in kws:
            values.append(kws.pop(name))
    if kws:
        raise TypeError("function has no keyword arguments: %s" %
                        tuple(kws.keys()))


def _make_docstring(name, decl):
    ret, args = decl
    doc = []

    doc.append("Wrapper for '%s'" % name)

    doc.append('')
    doc.append("Args")
    doc.append("----")
    for a, t in args:
        doc.append("%s: %s" % (a, t))

    return '\n'.join(doc)


def _init_api_function(name, decl):
    mangled = mangle(name)
    argnames = tuple([k for k, _ in decl[1]])
    fn = getattr(libcusparse, name)

    def method(self, *args, **kws):
        args = _flatten_args(args, kws, argnames)
        fn(self, *args)

    method.__doc__ = _make_docstring(name, decl)

    return mangled, method


def _init_cuSparse():
    gv = {}
    for k, v in _declarations():
        name, func = _init_api_function(k, v)
        assert name not in gv
        gv[name] = func

    # rewrite _v2 names
    for k in list(gv.keys()):
        if k.endswith('_v2'):
            stripped = k[:-3]
            assert stripped + '_v2' == k
            gv[stripped] = gv[k]

    base = _cuSparse
    return type('cuSparse', (base,), gv)


cuSparse = _init_cuSparse()
