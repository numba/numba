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
        try:
            self._api.cusparseCreate(byref(self._handle))
        except CuSparseError:
            raise RuntimeError("Cannot initialize cuSparse. "
                               "Could be caused by insufficient GPU memory.")
        self._finalizer_track((self._handle, self._api))
        # Default to NULL stream
        self._stream = 0
        # Default to host pointer
        self.use_host_pointer()

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

    @property
    def pointer_mode(self):
        mode = cusparsePointerMode_t()
        self._api.cusparseGetPointerMode(self._handle, byref(mode))
        return mode.value

    @pointer_mode.setter
    def pointer_mode(self, value):
        self._api.cusparseSetPointerMode(self._handle, value)

    def use_host_pointer(self):
        self.pointer_mode = CUSPARSE_POINTER_MODE_HOST

    def use_device_pointer(self):
        self.pointer_mode = CUSPARSE_POINTER_MODE_DEVICE


_strip_prefix = 'cusparse'
_len_strip_prefix = len(_strip_prefix)


def mangle(name):
    assert name.startswith(_strip_prefix)
    name = name[_len_strip_prefix:]
    return name


def _flatten_args(args, kws, argnames, defaults):
    values = list(args)
    for name in argnames[len(args):]:
        if name in kws:
            values.append(kws.pop(name))
        elif name in defaults:
            values.append(defaults[name])
        else:
            raise TypeError("missing '%s' arg" % name)
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


class _api_function(object):
    __slots__ = 'fn', 'argtypes', 'argnames', 'defaults'

    def __init__(self, fn, decl):
        self.fn = fn
        self.argnames, self.argtypes = zip(*decl[1])
        self.defaults = {}
        self.set_defaults()
        assert self.argnames[0] == 'handle'
        preparers = []
        for k in self.argnames:
            pname = 'prepare_%s' % k
            if hasattr(self, pname):
                meth = getattr(self, pname)
                preparers.append(meth)
            else:
                preparers.append(None)

        self.preparers = tuple(preparers)

    def __call__(self, *args, **kws):
        args = _flatten_args(args, kws, self.argnames, self.defaults)
        rargs = [pre(val) for pre, val in zip(self.preparers, args) if pre]
        actual, hold = zip(*rargs)
        self.fn(*args)
        return self.return_value(*hold)

    def set_defaults(self):
        if ('idxBase' in self.argnames and
                'cusparseIndexBase_t' in self.argtypes):
            self.defaults['idxBase'] = CUSPARSE_INDEX_BASE_ZERO

    def return_value(self, *args):
        return


def _make_api_function(name, base):
    return type(name, (base,), {})


def _prepare_array(self, val):
    return device_pointer(val), val


def _prepare_hybpartition(self, val):
    if val == 'A':
        return CUSPARSE_HYB_PARTITION_AUTO
    elif val == 'U':
        return CUSPARSE_HYB_PARTITION_USER
    elif val == 'M':
        return CUSPARSE_HYB_PARTITION_MAX
    else:
        raise ValueError("Partition flag must be either 'A', 'U' or 'M'")


def _prepare_direction_flag(self, val):
    if val == 'R':
        return CUSPARSE_DIRECTION_ROW
    elif val == 'C':
        return CUSPARSE_DIRECTION_COLUMN
    else:
        raise ValueError("Direction flag must be either 'R' or 'C'")


def _prepare_operation_flag(self, val):
    if val == 'N':
        return CUSPARSE_OPERATION_NON_TRANSPOSE
    elif val == 'T':
        return CUSPARSE_OPERATION_TRANSPOSE
    elif val == 'C':
        return CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE
    else:
        raise ValueError("Operation flag must be either 'N', 'T' or 'C'")


def _prepare_matdescr(self, val):
    raise NotImplementedError


def _prepare_hybmat(self, val):
    raise NotImplementedError


def _prepare_action(self, val):
    if val == 'N':
        return CUSPARSE_ACTION_NUMERIC
    elif val == 'S':
        return CUSPARSE_ACTION_SYMBOLIC
    else:
        raise ValueError("Action must be either 'N' or 'S'")


def _prepare_solveinfo(self, val):
    raise NotImplementedError


def _prepare_scalar(self, val):
    v = self.T(val)
    return byref(v), v


_prepare_scalar_out = _prepare_scalar

class _axpyi_v2(_api_function):
    __slots__ = ()

    prepare_alpha = _prepare_scalar
    prepare_xVal = _prepare_array
    prepare_xInt = _prepare_array
    prepare_y = _prepare_array


class Saxpyi_v2(_axpyi_v2):
    __slots__ = ()
    T = c_float


class Daxpyi_v2(_axpyi_v2):
    __slots__ = ()
    T = c_double


class Caxpyi_v2(_axpyi_v2):
    __slots__ = ()
    T = c_complex


class Zaxpyi_v2(_axpyi_v2):
    __slots__ = ()
    T = c_double_complex


class _bsr2csr(_api_function):
    __slots__ = ()

    prepare_dirA = _prepare_direction_flag

    prepare_bsrValA = _prepare_array
    prepare_bsrRowPtrA = _prepare_array
    prepare_bsrColIndA = _prepare_array
    prepare_csrValC = _prepare_array
    prepare_csrRowPtrC = _prepare_array
    prepare_csrColIndC = _prepare_array

    prepare_descrA = _prepare_matdescr
    prepare_descrC = _prepare_matdescr



Sbsr2csr = Dbsr2csr = Cbsr2csr = Zbsr2csr = _bsr2csr


class _bsrmv(_api_function):
    __slots__ = ()

    prepare_dirA = _prepare_direction_flag
    prepare_transA = _prepare_operation_flag
    prepare_alpha = _prepare_scalar
    prepare_beta = _prepare_scalar

    prepare_bsrValA = _prepare_array
    prepare_bsrRowPtrA = _prepare_array
    prepare_bsrColIndA = _prepare_array
    prepare_x = _prepare_array
    prepare_y = _prepare_array

    prepare_descrA = _prepare_matdescr


class Sbsrmv(_bsrmv):
    __slots__ = ()
    T = c_float


class Dbsrmv(_bsrmv):
    __slots__ = ()
    T = c_double


class Cbsrmv(_bsrmv):
    __slots__ = ()
    T = c_complex


class Zbsrmv(_bsrmv):
    __slots__ = ()
    T = c_double_complex


class _bsrxmv(_api_function):
    __slots__ = ()

    prepare_dirA = _prepare_direction_flag
    prepare_transA = _prepare_operation_flag

    prepare_alpha = _prepare_scalar
    prepare_beta = _prepare_scalar

    prepare_bsrMaskPtrA = _prepare_array
    prepare_bsrRowPtrA = _prepare_array
    prepare_bsrEndPtrA = _prepare_array
    prepare_bsrColIndA = _prepare_array

    prepare_x = _prepare_array
    prepare_y = _prepare_array

    prepare_descrA = _prepare_matdescr

Sbsrxmv = Dbsrxmv = Cbsrxmv = Zbsrxmv = _bsrxmv


class _csc2dense(_api_function):
    __slots__ = ()

    prepare_dirA = _prepare_direction_flag
    prepare_transA = _prepare_operation_flag

    prepare_alpha = _prepare_scalar
    prepare_beta = _prepare_scalar

    prepare_cscValA = _prepare_array
    prepare_cscRowIndA = _prepare_array
    prepare_cscColPtrA = _prepare_array
    prepare_A = _prepare_array

    prepare_descrA = _prepare_matdescr

Scsc2dense = Dcsc2dense = Ccsc2dense = Zcsc2dense = _csc2dense


class _csc2hyb(_api_function):
    __slots__ = ()
    prepare_descrA = _prepare_matdescr
    prepare_cscValA = _prepare_array
    prepare_cscRowIndA = _prepare_array
    prepare_cscColPtrA = _prepare_array
    prepare_hybA = _prepare_hybmat
    prepare_partitionType = _prepare_hybpartition

Scsc2hyb = Dcsc2hyb = Ccsc2hyb = Zcsc2hyb = _csc2hyb


class _csr2bsr(_api_function):
    __slots__ = ()

    prepare_dirA = _prepare_direction_flag
    prepare_descrA = _prepare_matdescr

    prepare_csrValA = _prepare_array
    prepare_csrRowPtrA = _prepare_array
    prepare_csrColIndA = _prepare_array

    prepare_descrC = _prepare_matdescr

    prepare_bsrValC = _prepare_array
    prepare_bsrRowPtrC = _prepare_array
    prepare_bsrColIndC = _prepare_array

Scsr2bsr = Dcsr2bsr = Ccsr2bsr = Zcsr2bsr = _csr2bsr


class _csr2csc_v2(_api_function):
    __slots__ = ()

    csrVal = _prepare_array
    csrRowPtr = _prepare_array
    csrColInd = _prepare_array
    cscVal = _prepare_array
    cscRowInd = _prepare_array
    cscColPtr = _prepare_array

    copyValues = _prepare_action

Scsr2csc_v2 = Dcsr2csc_v2 = Ccsr2csc_v2 = Zcsr2csc_v2 = _csr2csc_v2


class _csr2dense(_api_function):
    __slots__ = ()
    prepare_descrA = _prepare_matdescr
    prepare_csrValA = _prepare_array
    prepare_csrRowPtrA  = _prepare_array
    prepare_csrColIndA = _prepare_array
    prepare_A = _prepare_array

Scsr2dense = Dcsr2dense = Ccsr2dense = Zcsr2dense = _csr2dense


class _csr2hyb(_api_function):
    descrA = _prepare_matdescr
    csrValA = _prepare_array
    csrRowPtrA = _prepare_array
    csrColIndA = _prepare_array
    hybA = _prepare_hybmat
    partitionType = _prepare_hybpartition

Scsr2hyb = Dcsr2hyb = Ccsr2hyb = Zcsr2hyb = _csr2hyb


class _csrgeam(_api_function):

    prepare_alpha = _prepare_scalar
    prepare_beta = _prepare_scalar

    prepare_descrA = _prepare_matdescr
    prepare_csrValA = _prepare_array
    prepare_csrRowPtrA = _prepare_array
    prepare_csrColIndA = _prepare_array

    prepare_descrB = _prepare_matdescr
    prepare_csrValB = _prepare_array
    prepare_csrRowPtrB = _prepare_array
    prepare_csrColIndB = _prepare_array

    prepare_descrC = _prepare_matdescr
    prepare_csrValC = _prepare_array
    prepare_csrRowPtrC = _prepare_array
    prepare_csrColIndC = _prepare_array


class Scsrgeam(_csrgeam):
    T = c_float


class Dcsrgeam(_csrgeam):
    T = c_double


class Ccsrgeam(_csrgeam):
    T = c_complex


class Zcsrgeam(_csrgeam):
    T = c_double_complex


class _csrgemm(_api_function):
    prepare_transA = _prepare_action
    prepare_transB = _prepare_action
    prepare_descrA = _prepare_matdescr

    prepare_csrValA = _prepare_array
    prepare_csrRowPtrA = _prepare_array
    prepare_csrColIndA = _prepare_array
    prepare_descrB = _prepare_matdescr

    prepare_csrValB = _prepare_array
    prepare_csrRowPtrB = _prepare_array
    prepare_csrColIndB = _prepare_array
    prepare_descrC = _prepare_matdescr
    prepare_csrValC = _prepare_array
    prepare_csrRowPtrC = _prepare_array
    prepare_csrColIndC = _prepare_array

Scsrgemm = Dcsrgemm = Ccsrgemm = Zcsrgemm = _csrgemm


class _csric0(_api_function):
    prepare_trans = _prepare_operation_flag
    prepare_descrA = _prepare_matdescr
    prepare_csrValA_ValM = _prepare_array
    prepare_csrRowPtrA = _prepare_array
    prepare_csrColIndA = _prepare_array
    prepare_info = _prepare_solveinfo

Scsric0 = Dcsric0 = Ccsric0 = Zcsric0 = _csric0


class _csrilu0(_api_function):
    prepare_trans = _prepare_solveinfo
    prepare_descrA = _prepare_matdescr
    prepare_csrValA_ValM = _prepare_array
    prepare_csrRowPtrA = _prepare_array
    prepare_csrColIndA = _prepare_array
    prepare_info = _prepare_solveinfo

Scsrilu0 = Dcsrilu0 = Ccsrilu0 = Zcsrilu0 = _csrilu0


class _csrmm2(_api_function):
    prepare_transa = _prepare_action
    prepare_transb = _prepare_action

    prepare_alpha = _prepare_scalar
    prepare_beta = _prepare_scalar

    prepare_descrA = _prepare_matdescr

    prepare_csrValA = _prepare_array
    prepare_csrRowPtrA = _prepare_array
    prepare_csrColIndA = _prepare_array

    prepare_B = _prepare_array
    prepare_C = _prepare_array


class Scsrmm2(_csrmm2):
    T = c_float


class Dcsrmm2(_csrmm2):
    T = c_double


class Ccsrmm2(_csrmm2):
    T = c_complex


class Zcsrmm2(_csrmm2):
    T = c_double_complex


class _csrmm_v2(_api_function):
    prepare_transA = _prepare_operation_flag
    prepare_alpha = _prepare_scalar
    prepare_descrA = _prepare_matdescr
    prepare_csrValA = _prepare_array
    prepare_csrRowPtrA = _prepare_array
    prepare_csrColIndA = _prepare_array
    prepare_B = _prepare_array
    prepare_beta = _prepare_scalar
    prepare_C = _prepare_array


class Scsrmm_v2(_csrmm_v2):
    T = c_float


class Dcsrmm_v2(_csrmm_v2):
    T = c_double


class Ccsrmm_v2(_csrmm_v2):
    T = c_complex


class Zcsrmm_v2(_csrmm_v2):
    T = c_double_complex


class _csrmv_v2(_api_function):
    prepare_transA = _prepare_operation_flag
    prepare_alpha = _prepare_scalar
    prepare_descrA = _prepare_matdescr
    prepare_csrValA = _prepare_array
    prepare_csrRowPtrA = _prepare_array
    prepare_csrColIndA = _prepare_array
    prepare_x = _prepare_array
    prepare_beta = _prepare_scalar
    prepare_y = _prepare_array


class Scsrmv_v2(_csrmv_v2):
    T = c_float


class Dcsrmv_v2(_csrmv_v2):
    T = c_double


class Ccsrmv_v2(_csrmv_v2):
    T = c_complex


class Zcsrmv_v2(_csrmv_v2):
    T = c_double_complex


class _csrsm_analysis(_api_function):
    prepare_transA = _prepare_operation_flag
    prepare_descrA = _prepare_matdescr
    prepare_csrValA = _prepare_array
    prepare_csrRowPtrA = _prepare_array
    prepare_csrColIndA = _prepare_array
    prepare_info = _prepare_solveinfo

Scsrsm_analysis = Dcsrsm_analysis = _csrsm_analysis
Ccsrsm_analysis = Zcsrsm_analysis = _csrsm_analysis


class _csrsm_solve(_api_function):
    prepare_transA = _prepare_operation_flag
    prepare_alpha = _prepare_scalar
    prepare_descrA = _prepare_matdescr
    prepare_csrValA = _prepare_array
    prepare_csrRowPtrA = _prepare_array
    prepare_csrColIndA = _prepare_array
    prepare_info = _prepare_solveinfo
    prepare_x = _prepare_array
    prepare_y = _prepare_array

Scsrsm_solve = Dcsrsm_solve = Ccsrsm_solve = Zcsrsm_solve = _csrsm_solve


class _csrsv_analysis_v2(_api_function):
    prepare_transA = _prepare_operation_flag
    prepare_descrA = _prepare_matdescr
    prepare_csrValA = _prepare_array
    prepare_csrRowPtrA = _prepare_array
    prepare_csrColIndA = _prepare_array
    prepare_info = _prepare_solveinfo

Scsrsv_analysis_v2 = Dcsrsv_analysis_v2 = _csrsv_analysis_v2
Ccsrsv_analysis_v2 = Zcsrsv_analysis_v2 = _csrsv_analysis_v2


class _csrsv_solve_v2(_api_function):
    prepare_transA = _prepare_action
    prepare_alpha = _prepare_scalar
    prepare_descrA = _prepare_matdescr
    prepare_csrValA = _prepare_array
    prepare_csrRowPtrA = _prepare_array
    prepare_csrColIndA = _prepare_array
    prepare_info = _prepare_solveinfo
    prepare_x = _prepare_array
    prepare_y = _prepare_array

Scsrsv_solve_v2 = Dcsrsv_solve_v2 = _csrsv_solve_v2
Ccsrsv_solve_v2 = Zcsrsv_solve_v2 = _csrsv_solve_v2


class _dense2csc(_api_function):
    prepare_descrA = _prepare_matdescr
    prepare_A = _prepare_array
    prepare_nnzPerCol = _prepare_array
    prepare_cscValA = _prepare_array
    prepare_cscRowIndA = _prepare_array
    prepare_cscColPtrA = _prepare_array


Sdense2csc = Ddense2csc = Cdense2csc = Zdense2csc = _dense2csc


class _dense2csr(_api_function):
    prepare_descrA = _prepare_matdescr
    prepare_A = _prepare_array
    prepare_nnzPerRow = _prepare_array
    prepare_csrValA = _prepare_array
    prepare_csrRowPtrA = _prepare_array
    prepare_csrColIndA = _prepare_array

Sdense2csr = Ddense2csr = Cdense2csr = Zdense2csr = _dense2csr


class _dense2hyb(_api_function):
    prepare_descrA = _prepare_matdescr
    prepare_A = _prepare_array
    prepare_nnzPerRow = _prepare_array
    prepare_hybA = _prepare_hybmat
    prepare_partitionType = _prepare_hybpartition

Sdense2hyb = Ddense2hyb = Cdense2hyb = Zdense2hyb = _dense2hyb


class _dotci(_api_function):
    prepare_xVal = _prepare_array
    prepare_xInd = _prepare_array
    prepare_y = _prepare_array

    prepare_resultDevHostPtr = _prepare_scalar_out

    def return_value(self, *args):
        return args[self.argnames.index('resultDevHostPtr')]

    def set_defaults(self):
        super(_dotci, self).set_defaults()
        self.defaults['resultDevHostPtr'] = 0


class Cdotci(_dotci):
    T = c_complex


class Zdotci(_dotci):
    T = c_double_complex


class _doti(_api_function):
    prepare_xVal = _prepare_array
    prepare_xInd = _prepare_array
    prepare_y = _prepare_array
    prepare_resultDevHostPtr = _prepare_scalar_out

    def return_value(self, *args):
        return args[self.argnames.index('resultDevHostPtr')]

Sdoti = Ddoti = Cdoti = Zdoti = _doti


class _gthr(_api_function):
    prepare_y = _prepare_array
    prepare_xVal = _prepare_array
    prepare_xInd = _prepare_array

Sgthr = Dgthr = Cgthr = Zgthr = _gthr


class _gthrz(_api_function):
    prepare_y = _prepare_array
    prepare_xVal = _prepare_array
    prepare_xInd = _prepare_array

Sgthrz = Dgthrz = Cgthrz = Zgthrz = _gthrz


class _gtsv(_api_function):
    prepare_dl = _prepare_array
    prepare_d = _prepare_array
    prepare_du = _prepare_array
    prepare_B = _prepare_array

Sgtsv = Dgtsv = Cgtsv = Zgtsv = _gtsv


class _gtsvStridedBatch(_api_function):
    prepare_dl = _prepare_array
    prepare_d = _prepare_array
    prepare_du = _prepare_array
    prepare_x = _prepare_array

SgtsvStridedBatch = DgtsvStridedBatch = _gtsvStridedBatch
CgtsvStridedBatch = ZgtsvStridedBatch = _gtsvStridedBatch


class _gtsv_nopivot(_api_function):
    prepare_dl = _prepare_array
    prepare_d = _prepare_array
    prepare_du = _prepare_array
    prepare_B = _prepare_array

Sgtsv_nopivot = Dgtsv_nopivot = Cgtsv_nopivot = Zgtsv_nopivot = _gtsv_nopivot


class _hyb2csc(_api_function):
    prepare_descrA = _prepare_matdescr
    prepare_hybA = _prepare_hybmat
    prepare_cscVal = _prepare_array
    prepare_cscRowInd = _prepare_array
    prepare_cscColPtr = _prepare_array

Shyb2csc = Dhyb2csc = Chyb2csc = Zhyb2csc = _hyb2csc


class _hyb2csr(_api_function):
    prepare_descrA = _prepare_matdescr
    prepare_hybA = _prepare_hybmat
    prepare_csrValA = _prepare_array
    prepare_csrRowPtrA = _prepare_array
    prepare_csrColIndA = _prepare_array

Shyb2csr = Dhyb2csr = Chyb2csr = Zhyb2csr = _hyb2csr


class _hyb2dense(_api_function):
    prepare_descrA = _prepare_matdescr
    prepare_hybA = _prepare_hybmat
    prepare_A = _prepare_array

Shyb2dense = Dhyb2dense = Chyb2dense = Zhyb2dense = _hyb2dense


class _hybmv(_api_function):
    prepare_transA = _prepare_operation_flag
    prepare_alpha = _prepare_scalar
    prepare_descrA = _prepare_matdescr
    prepare_hybA = _prepare_hybmat
    prepare_x = _prepare_array
    prepare_beta = _prepare_scalar

Shybmv = Dhybmv = Chybmv = Zhybmv = _hybmv


class _hybsv_analysis(_api_function):
    prepare_transA = _prepare_operation_flag
    prepare_descrA = _prepare_matdescr
    prepare_hybA = _prepare_hybmat
    prepare_info = _prepare_solveinfo


Shybsv_analysis = Dhybsv_analysis = _hybsv_analysis
Chybsv_analysis = Zhybsv_analysis = _hybsv_analysis


class _hybsv_solve(_api_function):
    prepare_trans = _prepare_operation_flag
    prepare_alpha = _prepare_scalar
    prepare_descra = _prepare_matdescr
    prepare_hybA = _prepare_hybmat
    prepare_info = _prepare_solveinfo
    prepare_x = _prepare_array
    prepare_y = _prepare_array

Shybsv_solve = Dhybsv_solve = Chybsv_solve = Zhybsv_solve = _hybsv_solve


class _nnz(_api_function):
    prepare_dirA = _prepare_direction_flag
    prepare_descrA = _prepare_matdescr
    prepare_A = _prepare_array
    prepare_nnzPerRowCol = _prepare_array
    prepare_nnzTotalDevHostPtr = _prepare_scalar_out

    def return_value(self, *args):
        return args[self.argnames.index('nnzTotalDevHostPtr')]


Snnz = Dnnz = Cnnz = Znnz = _nnz


class _sctr(_api_function):
    prepare_xVal = _prepare_array
    prepare_xInd = _prepare_array
    prepare_y = _prepare_array

Ssctr = Dsctr = Csctr = Zsctr = _sctr


class _roti_v2(_api_function):
    prepare_xVal = _prepare_array
    prepare_xInd = _prepare_array
    prepare_y = _prepare_array
    prepare_c = _prepare_array
    prepare_s = _prepare_array


Sroti_v2 = Droti_v2 = _roti_v2


class Xcoo2csr(_api_function):
    prepare_cooRowInd = _prepare_array
    prepare_csrRowPtr = _prepare_array


class Xcsr2coo(_api_function):
    csrRowPtr = _prepare_array
    cooRowInd = _prepare_array


class Xcsr2bsrNnz(_api_function):
    prepare_dirA = _prepare_direction_flag
    prepare_descrA = _prepare_matdescr
    prepare_csrRowPtrA = _prepare_array
    prepare_csrColIndA = _prepare_array
    prepare_descrC = _prepare_array
    prepare_bsrRowPtrC = _prepare_array
    prepare_nnzTotalDevHostPtr = _prepare_scalar_out

    def return_value(self, *args):
        return args[self.argnames.index('nnzTotalDevHostPtr')]


class XcsrgeamNnz(_api_function):
    prepare_descrA = _prepare_matdescr

    prepare_csrRowPtrA = _prepare_array
    prepare_csrColIndA = _prepare_array
    prepare_descrB = _prepare_matdescr

    prepare_csrRowPtrB = _prepare_array
    prepare_csrColIndB = _prepare_array
    prepare_descrC = _prepare_matdescr
    prepare_csrRowPtrC = _prepare_array
    prepare_nnzTotalDevHostPtr = _prepare_scalar_out

    def return_value(self, *args):
        return args[self.argnames.index('nnzTotalDevHostPtr')]


class XcsrgemmNnz(_api_function):
    prepare_transA = _prepare_operation_flag
    prepare_transB = _prepare_operation_flag
    prepare_descrA = _prepare_matdescr
    prepare_csrRowPtrA = _prepare_array
    prepare_csrColIndA = _prepare_array
    prepare_descrB = _prepare_matdescr
    prepare_csrRowPtrB = _prepare_array
    prepare_csrColIndB = _prepare_array
    prepare_descrC = _prepare_matdescr
    prepare_csrRowPtrC = _prepare_array
    prepare_nnzTotalDevHostPtr = _prepare_scalar_out


def _init_api_function(name, decl):
    mangled = mangle(name)
    for k in globals().keys():
        if mangled.endswith(k):
            base = globals()[k]
            break
    else:
        print("missing", name)
        raise NotImplementedError(name)
        return mangled, None

    docs = _make_docstring(name, decl)
    cls = _make_api_function(name, base)

    fn = getattr(libcusparse, name)

    obj = cls(fn, decl)

    def method(self, *args, **kws):
        return obj(self._handle, *args, **kws)
    method.__doc__ = docs

    return mangled, method


_bypassed = frozenset('''
cusparseCreate
cusparseDestroy
cusparseCreateHybMat
cusparseCreateMatDescr
cusparseCreateSolveAnalysisInfo
cusparseDestroyHybMat
cusparseDestroyMatDescr
cusparseDestroySolveAnalysisInfo
cusparseGetMatDiagType
cusparseGetLevelInfo
cusparseGetMatFillMode
cusparseGetMatIndexBase
cusparseGetMatType
cusparseSetMatDiagType
cusparseSetMatFillMode
cusparseSetMatIndexBase
cusparseSetMatType
cusparseGetPointerMode
cusparseSetPointerMode
cusparseSetStream
cusparseGetVersion
'''.split())


def _init_cuSparse():
    gv = {}
    for k, v in _declarations():
        if k not in _bypassed:
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
