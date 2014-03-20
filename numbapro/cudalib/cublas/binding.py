from __future__ import absolute_import
import numpy as np
from ctypes import c_float, c_double, byref, c_int, Structure, c_void_p, POINTER

from numba.cuda.cudadrv.drvapi import cu_stream
from numba.cuda.cudadrv.driver import device_pointer, host_pointer
from numbapro.cudalib.libutils import Lib, ctype_function
from numbapro._utils import finalizer
from numbapro.cudalib.cctypes import c_double_complex, c_complex

INV_STATUS = dict(
    CUBLAS_STATUS_SUCCESS=0,
    CUBLAS_STATUS_NOT_INITIALIZED=1,
    CUBLAS_STATUS_ALLOC_FAILED=3,
    CUBLAS_STATUS_INVALID_VALUE=7,
    CUBLAS_STATUS_ARCH_MISMATCH=8,
    CUBLAS_STATUS_MAPPING_ERROR=11,
    CUBLAS_STATUS_EXECUTION_FAILED=13,
    CUBLAS_STATUS_INTERNAL_ERROR=14
)

STATUS = dict((v, k) for k, v in INV_STATUS.items())

cublasStatus_t = c_int

CUBLAS_FILL_MODE_LOWER = 0
CUBLAS_FILL_MODE_UPPER = 1

CUBLAS_FILL_MODE_MAP = {
    'L': CUBLAS_FILL_MODE_LOWER,
    'U': CUBLAS_FILL_MODE_UPPER,
}

cublasFillMode_t = c_int

CUBLAS_DIAG_NON_UNIT = 0
CUBLAS_DIAG_UNIT = 1

cublasDiagType_t = c_int

CUBLAS_DIAG_MAP = {
    True: CUBLAS_DIAG_UNIT,
    False: CUBLAS_DIAG_NON_UNIT,
}

CUBLAS_SIDE_LEFT = 0
CUBLAS_SIDE_RIGHT = 1

CUBLAS_SIDE_MAP = {
    'L': CUBLAS_SIDE_LEFT,
    'R': CUBLAS_SIDE_RIGHT,
}

cublasSideMode_t = c_int

CUBLAS_OP_N = 0
CUBLAS_OP_T = 1
CUBLAS_OP_C = 2

cublasOperation_t = c_int

CUBLAS_POINTER_MODE_HOST = 0
CUBLAS_POINTER_MODE_DEVICE = 1

cublasPointerMode_t = c_int

CUBLAS_ATOMICS_NOT_ALLOWED = 0
CUBLAS_ATOMICS_ALLOWED = 1

cublasAtomicsMode_t = c_int

cublasHandle_t = c_void_p # opaque handle

CUBLAS_OP_MAP = {'N': CUBLAS_OP_N,
                 'T': CUBLAS_OP_T,
                 'H': CUBLAS_OP_C, }


class CuBLASError(Exception):
    def __init__(self, code):
        super(CuBLASError, self).__init__(STATUS[code])


class libcublas(Lib):
    lib = 'cublas'
    ErrorType = CuBLASError

    cublasCreate_v2 = ctype_function(cublasStatus_t,
                                     POINTER(cublasHandle_t))  # handle

    cublasDestroy_v2 = ctype_function(cublasStatus_t,
                                      cublasHandle_t)   # handle

    cublasGetVersion_v2 = ctype_function(cublasStatus_t,
                                         cublasHandle_t, # handle
                                         POINTER(c_int)) # version

    cublasSetStream_v2 = ctype_function(cublasStatus_t,
                                        cublasHandle_t, # handle
                                        cu_stream)      # streamId

    cublasGetStream_v2 = ctype_function(cublasStatus_t,
                                        cublasHandle_t, # handle
                                        POINTER(cu_stream))      # streamId

    cublasGetPointerMode_v2 = ctype_function(cublasStatus_t,
                                             cublasHandle_t, # handle
                                             POINTER(
                                                 cublasPointerMode_t)) # mode

    cublasSetPointerMode_v2 = ctype_function(cublasStatus_t,
                                             cublasHandle_t, # handle
                                             cublasPointerMode_t) # mode

    cublasGetAtomicsMode = ctype_function(cublasStatus_t,
                                          cublasHandle_t, # handle
                                          POINTER(cublasAtomicsMode_t)) # mode

    cublasSetAtomicsMode = ctype_function(cublasStatus_t,
                                          cublasHandle_t, # handle
                                          cublasAtomicsMode_t)  # mode

    # Level 1
    cublasSnrm2_v2 = ctype_function(cublasStatus_t,
                                    cublasHandle_t, # handle
                                    c_int, # n
                                    c_void_p, # device array
                                    c_int, # incx
                                    c_void_p)       # result - host/device scalar

    cublasDnrm2_v2 = cublasSnrm2_v2
    cublasScnrm2_v2 = cublasSnrm2_v2
    cublasDznrm2_v2 = cublasSnrm2_v2

    cublasSdot_v2 = ctype_function(cublasStatus_t,
                                   cublasHandle_t, # handle
                                   c_int, # n
                                   c_void_p, # x
                                   c_int, # incx
                                   c_void_p, # y
                                   c_int, # incy,
                                   c_void_p)   # result h/d ptr
    cublasDdot_v2 = cublasSdot_v2
    cublasCdotu_v2 = cublasSdot_v2
    cublasZdotu_v2 = cublasSdot_v2
    cublasCdotc_v2 = cublasSdot_v2
    cublasZdotc_v2 = cublasSdot_v2

    cublasSscal_v2 = ctype_function(cublasStatus_t,
                                    cublasHandle_t, # handle
                                    c_int, # n
                                    c_void_p, # alpha h/d
                                    c_void_p, # x
                                    c_int)            # incx

    cublasDscal_v2 = cublasSscal_v2
    cublasCscal_v2 = cublasSscal_v2
    cublasZscal_v2 = cublasSscal_v2
    cublasCsscal_v2 = cublasSscal_v2
    cublasZdscal_v2 = cublasSscal_v2

    cublasSaxpy_v2 = ctype_function(cublasStatus_t,
                                    cublasHandle_t, # handle
                                    c_int, # n
                                    c_void_p, # alpha h/d
                                    c_void_p, # x
                                    c_int, # incx
                                    c_void_p, # y
                                    c_int)          # incy
    cublasDaxpy_v2 = cublasSaxpy_v2
    cublasCaxpy_v2 = cublasSaxpy_v2
    cublasZaxpy_v2 = cublasSaxpy_v2

    cublasIsamax_v2 = ctype_function(cublasStatus_t,
                                     cublasHandle_t, # handle
                                     c_int, # n
                                     c_void_p, # x
                                     c_int, # incx
                                     POINTER(c_int)) # result h/d ptr

    cublasIdamax_v2 = cublasIsamax_v2
    cublasIcamax_v2 = cublasIsamax_v2
    cublasIzamax_v2 = cublasIsamax_v2

    cublasIsamin_v2 = cublasIsamax_v2

    cublasIdamin_v2 = cublasIsamin_v2
    cublasIcamin_v2 = cublasIsamin_v2
    cublasIzamin_v2 = cublasIsamin_v2

    cublasSasum_v2 = ctype_function(cublasStatus_t,
                                    cublasHandle_t, # handle
                                    c_int, # n
                                    c_void_p, # x
                                    c_int, # incx
                                    c_void_p)       # result h/d ptr

    cublasDasum_v2 = cublasSasum_v2
    cublasScasum_v2 = cublasSasum_v2
    cublasDzasum_v2 = cublasSasum_v2

    cublasSrot_v2 = ctype_function(cublasStatus_t,
                                   cublasHandle_t, # handle,
                                   c_int, # n
                                   c_void_p, # x
                                   c_int, # incx
                                   c_void_p, # y
                                   c_int, # incy
                                   c_void_p, # c
                                   c_void_p)        # s h/d ptr

    cublasDrot_v2 = cublasSrot_v2
    cublasCrot_v2 = cublasSrot_v2
    cublasZrot_v2 = cublasSrot_v2
    cublasCsrot_v2 = cublasSrot_v2
    cublasZdrot_v2 = cublasSrot_v2

    cublasSrotg_v2 = ctype_function(cublasStatus_t,
                                    cublasHandle_t, # handle,
                                    c_void_p, # a h/d ptr
                                    c_void_p, # b h/d ptr
                                    c_void_p, # c h/d ptr
                                    c_void_p)           # s h/d ptr

    cublasDrotg_v2 = cublasSrotg_v2
    cublasCrotg_v2 = cublasSrotg_v2
    cublasZrotg_v2 = cublasSrotg_v2

    cublasSrotm_v2 = ctype_function(cublasStatus_t,
                                    cublasHandle_t, # handle
                                    c_int, # n
                                    c_void_p, # x
                                    c_int, # incx
                                    c_void_p, # y
                                    c_int, # incy
                                    c_void_p)           # param h/d pointer
    cublasDrotm_v2 = cublasSrotm_v2

    cublasSrotmg_v2 = ctype_function(cublasStatus_t,
                                     cublasHandle_t, # handle,
                                     c_void_p, # d1 h/d ptr
                                     c_void_p, # d2 h/d ptr
                                     c_void_p, # x1 h/d ptr
                                     c_void_p, # y1 h/d ptr
                                     c_void_p)       # param h/d ptr

    cublasDrotmg_v2 = cublasSrotmg_v2

    #
    # Level 2
    #
    cublasSgbmv_v2 = ctype_function(cublasStatus_t,
                                    cublasHandle_t, # handle,
                                    cublasOperation_t, # trans,
                                    c_int, # m,
                                    c_int, # n,
                                    c_int, # kl,
                                    c_int, # ku,
                                    c_void_p, # *alpha,
                                    c_void_p, # *A,
                                    c_int, # lda,
                                    c_void_p, # *x,
                                    c_int, # incx,
                                    c_void_p, # *beta,
                                    c_void_p, # *y,
                                    c_int)             # incy)

    cublasDgbmv_v2 = cublasSgbmv_v2
    cublasCgbmv_v2 = cublasSgbmv_v2
    cublasZgbmv_v2 = cublasSgbmv_v2

    cublasSgemv_v2 = ctype_function(cublasStatus_t,
                                    cublasHandle_t, # handle,
                                    cublasOperation_t, # trans,
                                    c_int, # m,
                                    c_int, # n,
                                    c_void_p, # *alpha,
                                    c_void_p, # *A,
                                    c_int, # lda,
                                    c_void_p, # *x,
                                    c_int, # incx,
                                    c_void_p, # *beta,
                                    c_void_p, # *y,
                                    c_int)                 # incy)

    cublasDgemv_v2 = cublasSgemv_v2
    cublasCgemv_v2 = cublasSgemv_v2
    cublasZgemv_v2 = cublasSgemv_v2

    cublasStrmv_v2 = ctype_function(cublasStatus_t,
                                    cublasHandle_t, # handle,
                                    cublasFillMode_t, # uplo,
                                    cublasOperation_t, # trans,
                                    cublasDiagType_t, # diag,
                                    c_int, # n,
                                    c_void_p, # *A,
                                    c_int, # lda,
                                    c_void_p, # *x,
                                    c_int)             # incx);

    cublasDtrmv_v2 = cublasStrmv_v2
    cublasCtrmv_v2 = cublasStrmv_v2
    cublasZtrmv_v2 = cublasStrmv_v2

    cublasStbmv_v2 = ctype_function(cublasStatus_t,
                                    cublasHandle_t, # handle,
                                    cublasFillMode_t, # uplo,
                                    cublasOperation_t, # trans,
                                    cublasDiagType_t, # diag,
                                    c_int, # n,
                                    c_int, # k,
                                    c_void_p, # *A,
                                    c_int, # lda,
                                    c_void_p, # *x,
                                    c_int)                    # incx);

    cublasDtbmv_v2 = cublasStbmv_v2
    cublasCtbmv_v2 = cublasStbmv_v2
    cublasZtbmv_v2 = cublasStbmv_v2

    cublasStpmv_v2 = ctype_function(cublasStatus_t,
                                    cublasHandle_t, # handle,
                                    cublasFillMode_t, # uplo,
                                    cublasOperation_t, # trans,
                                    cublasDiagType_t, # diag,
                                    c_int, # n,
                                    c_void_p, # *AP,
                                    c_void_p, # *x,
                                    c_int)              # incx);

    cublasDtpmv_v2 = cublasStpmv_v2
    cublasCtpmv_v2 = cublasStpmv_v2
    cublasZtpmv_v2 = cublasStpmv_v2

    cublasStrsv_v2 = ctype_function(cublasStatus_t,
                                    cublasHandle_t, # handle,
                                    cublasFillMode_t, # uplo,
                                    cublasOperation_t, # trans,
                                    cublasDiagType_t, # diag,
                                    c_int, # n,
                                    c_void_p, # *A,
                                    c_int, # lda,
                                    c_void_p, # *x,
                                    c_int)                # incx);

    cublasDtrsv_v2 = cublasStrsv_v2
    cublasCtrsv_v2 = cublasStrsv_v2
    cublasZtrsv_v2 = cublasStrsv_v2

    cublasStpsv_v2 = ctype_function(cublasStatus_t,
                                    cublasHandle_t, #handle,
                                    cublasFillMode_t, #uplo,
                                    cublasOperation_t, #trans,
                                    cublasDiagType_t, #diag,
                                    c_int, #n,
                                    c_void_p, #*AP,
                                    c_void_p, #*x,
                                    c_int)              #incx);

    cublasDtpsv_v2 = cublasStpsv_v2
    cublasCtpsv_v2 = cublasStpsv_v2
    cublasZtpsv_v2 = cublasStpsv_v2

    cublasStbsv_v2 = ctype_function(cublasStatus_t,
                                    cublasHandle_t, #handle,
                                    cublasFillMode_t, #uplo,
                                    cublasOperation_t, #trans,
                                    cublasDiagType_t, #diag,
                                    c_int, #n,
                                    c_int, #k,
                                    c_void_p, #*A,
                                    c_int, #lda,
                                    c_void_p, #*x,
                                    c_int)              #incx);

    cublasDtbsv_v2 = cublasStbsv_v2
    cublasCtbsv_v2 = cublasStbsv_v2
    cublasZtbsv_v2 = cublasStbsv_v2

    cublasSsymv_v2 = ctype_function(cublasStatus_t,
                                    cublasHandle_t, #handle,
                                    cublasFillMode_t, #uplo,
                                    c_int, #n,
                                    c_void_p, #*alpha,
                                    c_void_p, #*A,
                                    c_int, #lda,
                                    c_void_p, #*x,
                                    c_int, #incx,
                                    c_void_p, #*beta,
                                    c_void_p, #*y,
                                    c_int)              #incy);

    cublasDsymv_v2 = cublasSsymv_v2
    cublasCsymv_v2 = cublasSsymv_v2
    cublasZsymv_v2 = cublasSsymv_v2

    cublasChemv_v2 = ctype_function(cublasStatus_t,
                                    cublasHandle_t, #handle,
                                    cublasFillMode_t, #uplo,
                                    c_int, #n,
                                    c_void_p, #*alpha,
                                    c_void_p, #*A,
                                    c_int, #lda,
                                    c_void_p, #*x,
                                    c_int, #incx,
                                    c_void_p, #*beta,
                                    c_void_p, #*y,
                                    c_int)             #incy);
    cublasZhemv_v2 = cublasChemv_v2

    cublasSsbmv_v2 = ctype_function(cublasStatus_t,
                                    cublasHandle_t, #handle,
                                    cublasFillMode_t, #uplo,
                                    c_int, #n,
                                    c_int, #k,
                                    c_void_p, #*alpha
                                    c_void_p, #*A,
                                    c_int, #lda,
                                    c_void_p, #*x,
                                    c_int, #incx,
                                    c_void_p, #*beta
                                    c_void_p, #*y,
                                    c_int)             #incy);
    cublasDsbmv_v2 = cublasSsbmv_v2

    cublasChbmv_v2 = ctype_function(cublasStatus_t,
                                    cublasHandle_t, #handle,
                                    cublasFillMode_t, #uplo,
                                    c_int, #n,
                                    c_int, #k,
                                    c_void_p, #*alpha,
                                    c_void_p, #*A,
                                    c_int, #lda,
                                    c_void_p, #*x,
                                    c_int, #incx,
                                    c_void_p, #*beta,
                                    c_void_p, #*y,
                                    c_int)                 #incy);
    cublasZhbmv_v2 = cublasChbmv_v2

    cublasSspmv_v2 = ctype_function(cublasStatus_t,
                                    cublasHandle_t, #handle,
                                    cublasFillMode_t, #uplo,
                                    c_int, #n,
                                    c_void_p, #*alpha,
                                    c_void_p, #*AP,
                                    c_void_p, #*x,
                                    c_int, #incx,
                                    c_void_p, #*beta,
                                    c_void_p, #*y,
                                    c_int)     #incy);

    cublasDspmv_v2 = cublasSspmv_v2
    cublasChpmv_v2 = cublasSspmv_v2
    cublasZhpmv_v2 = cublasChpmv_v2

    cublasSger_v2 = ctype_function(cublasStatus_t,
                                   cublasHandle_t, #handle,
                                   c_int, #m,
                                   c_int, #n,
                                   c_void_p, #*alpha,
                                   c_void_p, #*x,
                                   c_int, #incx,
                                   c_void_p, #*y,
                                   c_int, #incy,
                                   c_void_p, #*A,
                                   c_int)           #lda);
    cublasDger_v2 = cublasSger_v2
    cublasCgeru_v2 = cublasDger_v2
    cublasCgerc_v2 = cublasDger_v2
    cublasZgeru_v2 = cublasDger_v2
    cublasZgerc_v2 = cublasDger_v2

    cublasSsyr_v2 = ctype_function(cublasStatus_t,
                                   cublasHandle_t, #handle,
                                   cublasFillMode_t, #uplo,
                                   c_int, #n,
                                   c_void_p, #*alpha,
                                   c_void_p, #*x,
                                   c_int, #incx,
                                   c_void_p, #*A,
                                   c_int)              #lda);
    cublasDsyr_v2 = cublasSsyr_v2
    cublasCsyr_v2 = cublasSsyr_v2
    cublasZsyr_v2 = cublasSsyr_v2

    cublasCher_v2 = ctype_function(cublasStatus_t,
                                   cublasHandle_t, #handle,
                                   cublasFillMode_t, #uplo,
                                   c_int, #n,
                                   c_void_p, #*alpha,
                                   c_void_p, #*x,
                                   c_int, #incx,
                                   c_void_p, #*A,
                                   c_int)       #lda);

    cublasZher_v2 = cublasCher_v2

    cublasSspr_v2 = ctype_function(cublasStatus_t,
                                   cublasHandle_t, #handle,
                                   cublasFillMode_t, # uplo,
                                   c_int, #n,
                                   c_void_p, #*alpha,
                                   c_void_p, #*x,
                                   c_int, #incx,
                                   c_void_p)         #*AP);

    cublasDspr_v2 = cublasSspr_v2
    cublasChpr_v2 = cublasSspr_v2
    cublasZhpr_v2 = cublasSspr_v2

    cublasSsyr2_v2 = ctype_function(cublasStatus_t,
                                    cublasHandle_t, #handle,
                                    cublasFillMode_t, #uplo,
                                    c_int, #n,
                                    c_void_p, #*alpha,
                                    c_void_p, #*x,
                                    c_int, #incx,
                                    c_void_p, #*y,
                                    c_int, #incy,
                                    c_void_p, #*A,
                                    c_int)                 #lda);

    cublasDsyr2_v2 = cublasSsyr2_v2
    cublasCsyr2_v2 = cublasSsyr2_v2
    cublasZsyr2_v2 = cublasSsyr2_v2
    cublasCher2_v2 = cublasSsyr2_v2
    cublasZher2_v2 = cublasSsyr2_v2

    cublasSspr2_v2 = ctype_function(cublasStatus_t,
                                    cublasHandle_t, #handle,
                                    cublasFillMode_t, #uplo,
                                    c_int, #n,
                                    c_void_p, #*alpha
                                    c_void_p, #*x,
                                    c_int, #incx,
                                    c_void_p, #*y,
                                    c_int, #incy,
                                    c_void_p)          #*AP);

    cublasDspr2_v2 = cublasSspr2_v2

    cublasChpr2_v2 = cublasSspr2_v2
    cublasZhpr2_v2 = cublasSspr2_v2

    # Level 3
    cublasSgemm_v2 = ctype_function(cublasStatus_t,
                                    cublasHandle_t, #handle,
                                    cublasOperation_t, #transa,
                                    cublasOperation_t, #transb,
                                    c_int, #m,
                                    c_int, #n,
                                    c_int, #k,
                                    c_void_p, #*alpha
                                    c_void_p, #*A,
                                    c_int, #lda,
                                    c_void_p, #*B,
                                    c_int, #ldb,
                                    c_void_p, #*beta,
                                    c_void_p, #*C,
                                    c_int)         #ldc);

    cublasDgemm_v2 = cublasSgemm_v2
    cublasCgemm_v2 = cublasSgemm_v2
    cublasZgemm_v2 = cublasSgemm_v2

    cublasSsyrk_v2 = ctype_function(cublasStatus_t,
                                    cublasHandle_t, #handle,
                                    cublasFillMode_t, #uplo,
                                    cublasOperation_t, #trans,
                                    c_int, #n,
                                    c_int, #k,
                                    c_void_p, #*alpha,
                                    c_void_p, #*A,
                                    c_int, #lda,
                                    c_void_p, #*beta,
                                    c_void_p, #*C,
                                    c_int)             #ldc);

    cublasDsyrk_v2 = cublasSsyrk_v2
    cublasCsyrk_v2 = cublasSsyrk_v2
    cublasZsyrk_v2 = cublasSsyrk_v2

    cublasCherk_v2 = ctype_function(cublasStatus_t,
                                    cublasHandle_t, #handle,
                                    cublasFillMode_t, #uplo,
                                    cublasOperation_t, #trans,
                                    c_int, #n,
                                    c_int, #k,
                                    c_void_p, #*alpha,
                                    c_void_p, #*A,
                                    c_int, #lda,
                                    c_void_p, #*beta
                                    c_void_p, #*C,
                                    c_int)                 #ldc);
    cublasZherk_v2 = cublasCherk_v2

    cublasSsymm_v2 = ctype_function(cublasStatus_t,
                                    cublasHandle_t, #handle,
                                    cublasSideMode_t, #side,
                                    cublasFillMode_t, #uplo,
                                    c_int, #m,
                                    c_int, #n,
                                    c_void_p, #*alpha
                                    c_void_p, #*A,
                                    c_int, #lda,
                                    c_void_p, #*B,
                                    c_int, #ldb,
                                    c_void_p, #*beta
                                    c_void_p, #*C,
                                    c_int)             #ldc);

    cublasDsymm_v2 = cublasSsymm_v2
    cublasCsymm_v2 = cublasSsymm_v2
    cublasZsymm_v2 = cublasSsymm_v2

    cublasChemm_v2 = ctype_function(cublasStatus_t,
                                    cublasHandle_t, #handle,
                                    cublasSideMode_t, #side,
                                    cublasFillMode_t, #uplo,
                                    c_int, #m,
                                    c_int, #n,
                                    c_void_p, #*alpha
                                    c_void_p, #*A,
                                    c_int, #lda,
                                    c_void_p, #*B,
                                    c_int, #ldb,
                                    c_void_p, #*beta
                                    c_void_p, #*C,
                                    c_int)             #ldc);
    cublasZhemm_v2 = cublasChemm_v2

    cublasStrsm_v2 = ctype_function(cublasStatus_t,
                                    cublasHandle_t, #handle,
                                    cublasSideMode_t, #side,
                                    cublasFillMode_t, #uplo,
                                    cublasOperation_t, #trans,
                                    cublasDiagType_t, #diag,
                                    c_int, #m,
                                    c_int, #n,
                                    c_void_p, #*alpha
                                    c_void_p, #*A,
                                    c_int, #lda,
                                    c_void_p, #*B,
                                    c_int)                 #ldb);

    cublasDtrsm_v2 = cublasStrsm_v2
    cublasCtrsm_v2 = cublasStrsm_v2
    cublasZtrsm_v2 = cublasStrsm_v2

    cublasStrmm_v2 = ctype_function(cublasStatus_t,
                                    cublasHandle_t, #handle,
                                    cublasSideMode_t, #side,
                                    cublasFillMode_t, #uplo,
                                    cublasOperation_t, #trans,
                                    cublasDiagType_t, #diag,
                                    c_int, #m,
                                    c_int, #n,
                                    c_void_p, #*alpha
                                    c_void_p, #*A,
                                    c_int, #lda,
                                    c_void_p, #*B,
                                    c_int, #ldb,
                                    c_void_p, #*C,
                                    c_int)              #ldc);

    cublasDtrmm_v2 = cublasStrmm_v2
    cublasCtrmm_v2 = cublasStrmm_v2
    cublasZtrmm_v2 = cublasStrmm_v2

    cublasSdgmm = ctype_function(cublasStatus_t,
                                 cublasHandle_t, #handle,
                                 cublasSideMode_t, #mode,
                                 c_int, #m,
                                 c_int, #n,
                                 c_void_p, #*A,
                                 c_int, #lda,
                                 c_void_p, #*x,
                                 c_int, #incx,
                                 c_void_p, #*C,
                                 c_int)             #ldc);
    cublasDdgmm = cublasSdgmm
    cublasCdgmm = cublasSdgmm
    cublasZdgmm = cublasSdgmm

    cublasSgeam = ctype_function(cublasStatus_t,
                                 cublasHandle_t, #handle,
                                 cublasOperation_t, #transa,
                                 cublasOperation_t, #transb,
                                 c_int, #m,
                                 c_int, #n,
                                 c_void_p, #*alpha,
                                 c_void_p, #*A,
                                 c_int, #lda,
                                 c_void_p, #*beta,
                                 c_void_p, #*B,
                                 c_int, #ldb,
                                 c_void_p, #*C
                                 c_int)                 #ldc);
    cublasDgeam = cublasSgeam
    cublasCgeam = cublasSgeam
    cublasZgeam = cublasSgeam


def _return_scalar(result):
    if isinstance(result, (c_complex, c_double_complex)):
        return complex(result.real, result.imag)
    else:
        return result.value


def _Tnrm2(fmt, cty):
    def nrm2(self, n, x, incx):
        result = cty()
        fn = getattr(self._api, 'cublas%snrm2_v2' % fmt)
        fn(self._handle, int(n), device_pointer(x), int(incx),
           byref(result))
        return _return_scalar(result)

    return nrm2


def _Tdot(fmt, cty, postfix=''):
    def dot(self, n, x, incx, y, incy):
        result = cty()
        fn = getattr(self._api, 'cublas%sdot%s_v2' % (fmt, postfix))
        fn(self._handle, int(n), device_pointer(x), int(incx),
           device_pointer(y), int(incy), byref(result))
        return _return_scalar(result)

    return dot


def _Tscal(fmt, cty):
    def scal(self, n, alpha, x, incx):
        "Stores result to x"
        c_alpha = cty(alpha)
        fn = getattr(self._api, 'cublas%sscal_v2' % fmt)
        fn(self._handle, int(n), byref(c_alpha), device_pointer(x),
           int(incx))

    return scal


def _Taxpy(fmt, cty):
    def axpy(self, n, alpha, x, incx, y, incy):
        "Stores result to y"
        c_alpha = cty(alpha)
        fn = getattr(self._api, 'cublas%saxpy_v2' % fmt)
        fn(self._handle, int(n), byref(c_alpha), device_pointer(x),
           int(incx), device_pointer(y), int(incy))

    return axpy


def _Itamax(fmt, cty):
    def amax(self, n, x, incx):
        result = c_int()
        fn = getattr(self._api, 'cublasI%samax_v2' % fmt)
        fn(self._handle, int(n), device_pointer(x), int(incx),
           byref(result))
        return result.value

    return amax


def _Itamin(fmt, cty):
    def amin(self, n, x, incx):
        result = c_int()
        fn = getattr(self._api, 'cublasI%samin_v2' % fmt)
        fn(self._handle, int(n), device_pointer(x), int(incx),
           byref(result))
        return result.value

    return amin


def _Tasum(fmt, cty):
    def asum(self, n, x, incx):
        result = cty()
        fn = getattr(self._api, 'cublas%sasum_v2' % fmt)
        fn(self._handle, int(n), device_pointer(x), int(incx),
           byref(result))
        return _return_scalar(result)

    return asum


def _Trot(fmt, cty, sty):
    def rot(self, n, x, incx, y, incy, c, s):
        "Stores to x and y"
        c_c = cty(c)
        c_s = sty(s)
        fn = getattr(self._api, 'cublas%srot_v2' % fmt)
        fn(self._handle, int(n), device_pointer(x), int(incx),
           device_pointer(y), int(incy), byref(c_c), byref(c_s))

    return rot


def _Trotg(fmt, ty, cty):
    def rotg(self, a, b):
        c_a = ty(a)
        c_b = ty(b)
        c_c = cty()
        c_s = ty()
        fn = getattr(self._api, 'cublas%srotg_v2' % fmt)
        fn(self._handle, byref(c_a), byref(c_b), byref(c_c), byref(c_s))
        r, z, c, s = map(_return_scalar, [c_a, c_b, c_c, c_s])
        return r, z, c, s

    return rotg


def _Trotm(fmt, dtype):
    def rotm(self, n, x, incx, y, incy, param):
        "Stores result to x, y"
        fn = getattr(self._api, 'cublas%srotm_v2' % fmt)
        assert len(param.shape) == 1, "param must be a 1-d array"
        assert param.size >= 5, "param must have at least 5 elements"
        assert param.dtype == np.dtype(dtype), "param dtype mismatch"
        fn(self._handle, int(n), device_pointer(x), int(incx),
           device_pointer(y), int(incy), host_pointer(param))

    return rotm


def _Trotmg(fmt, cty, dtype):
    def rotmg(self, d1, d2, x1, y1):
        fn = getattr(self._api, 'cublas%srotmg_v2' % fmt)
        c_d1 = cty(d1)
        c_d2 = cty(d2)
        c_x1 = cty(x1)
        c_y1 = cty(y1)
        param = np.zeros(5, dtype=dtype)
        fn(self._handle, byref(c_d1), byref(c_d2), byref(c_x1), byref(c_y1),
           host_pointer(param))
        return param

    return rotmg


def _Tgbmv(fmt, cty, dtype):
    def gbmv(self, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy):
        '''This function performs the banded matrix-vector multiplication
        '''
        fn = getattr(self._api, 'cublas%sgbmv_v2' % fmt)
        c_alpha = cty(alpha)
        c_beta = cty(beta)
        trans = CUBLAS_OP_MAP[trans]
        fn(self._handle, trans, m, n, kl, ku, byref(c_alpha), device_pointer(A),
           lda, device_pointer(x), incx, byref(c_beta), device_pointer(y), incy)

    return gbmv


def _Tgemv(fmt, cty, dtype):
    def gemv(self, trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
        '''This function performs the banded matrix-vector multiplication
        '''
        fn = getattr(self._api, 'cublas%sgemv_v2' % fmt)
        c_alpha = cty(alpha)
        c_beta = cty(beta)
        trans = {'N': CUBLAS_OP_N,
                 'T': CUBLAS_OP_T,
                 'H': CUBLAS_OP_C, }[trans]
        fn(self._handle, trans, m, n, byref(c_alpha), device_pointer(A),
           lda, device_pointer(x), incx, byref(c_beta), device_pointer(y), incy)

    return gemv


def _Ttrmv(fmt, dtype):
    def trmv(self, uplo, trans, diag, n, A, lda, x, incx):
        fn = getattr(self._api, 'cublas%strmv_v2' % fmt)
        fn(self._handle, CUBLAS_FILL_MODE_MAP[uplo], CUBLAS_OP_MAP[trans],
           CUBLAS_DIAG_MAP[diag], n, device_pointer(A), lda, device_pointer(x),
           incx)

    return trmv


def _Ttbmv(fmt, dtype):
    def tbmv(self, uplo, trans, diag, n, k, A, lda, x, incx):
        fn = getattr(self._api, 'cublas%stbmv_v2' % fmt)
        fn(self._handle, CUBLAS_FILL_MODE_MAP[uplo], CUBLAS_OP_MAP[trans],
           CUBLAS_DIAG_MAP[diag], n, k, device_pointer(A), lda,
           device_pointer(x), incx)

    return tbmv


def _Ttpmv(fmt, dtype):
    def tpmv(self, uplo, trans, diag, n, AP, x, incx):
        fn = getattr(self._api, 'cublas%stpmv_v2' % fmt)
        fn(self._handle, CUBLAS_FILL_MODE_MAP[uplo], CUBLAS_OP_MAP[trans],
           CUBLAS_DIAG_MAP[diag], n, device_pointer(AP), device_pointer(x),
           incx)

    return tpmv


def _Ttrsv(fmt, dtype):
    def trsv(self, uplo, trans, diag, n, A, lda, x, incx):
        fn = getattr(self._api, 'cublas%strsv_v2' % fmt)
        fn(self._handle, CUBLAS_FILL_MODE_MAP[uplo], CUBLAS_OP_MAP[trans],
           CUBLAS_DIAG_MAP[diag], n, device_pointer(A), lda, device_pointer(x),
           incx)

    return trsv


def _Ttpsv(fmt, dtype):
    def tpsv(self, uplo, trans, diag, n, AP, x, incx):
        fn = getattr(self._api, 'cublas%stpsv_v2' % fmt)
        fn(self._handle, CUBLAS_FILL_MODE_MAP[uplo], CUBLAS_OP_MAP[trans],
           CUBLAS_DIAG_MAP[diag], n, device_pointer(AP), device_pointer(x),
           incx)

    return tpsv


def _Ttbsv(fmt, dtype):
    def tbsv(self, uplo, trans, diag, n, k, A, lda, x, incx):
        fn = getattr(self._api, 'cublas%stbsv_v2' % fmt)
        fn(self._handle, CUBLAS_FILL_MODE_MAP[uplo], CUBLAS_OP_MAP[trans],
           CUBLAS_DIAG_MAP[diag], n, k, device_pointer(A), lda,
           device_pointer(x), incx)

    return tbsv


def _Tsymv(fmt, cty, dtype):
    def symv(self, uplo, n, alpha, A, lda, x, incx, beta, y, incy):
        fn = getattr(self._api, 'cublas%ssymv_v2' % fmt)
        c_alpha = cty(alpha)
        c_beta = cty(beta)
        fn(self._handle, CUBLAS_FILL_MODE_MAP[uplo], n, byref(c_alpha),
           device_pointer(A), lda, device_pointer(x), incx, byref(c_beta),
           device_pointer(y), incy)

    return symv


def _Themv(fmt, cty, dtype):
    def symv(self, uplo, n, alpha, A, lda, x, incx, beta, y, incy):
        fn = getattr(self._api, 'cublas%shemv_v2' % fmt)
        c_alpha = cty(alpha)
        c_beta = cty(beta)
        fn(self._handle, CUBLAS_FILL_MODE_MAP[uplo], n, byref(c_alpha),
           device_pointer(A), lda, device_pointer(x), incx, byref(c_beta),
           device_pointer(y), incy)

    return symv


def _Tsbmv(fmt, cty, dtype):
    def sbmv(self, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy):
        fn = getattr(self._api, 'cublas%ssbmv_v2' % fmt)
        c_alpha = cty(alpha)
        c_beta = cty(beta)
        fn(self._handle, CUBLAS_FILL_MODE_MAP[uplo], n, k, byref(c_alpha),
           device_pointer(A), lda, device_pointer(x), incx, byref(c_beta),
           device_pointer(y), incy)

    return sbmv


def _Thbmv(fmt, cty, dtype):
    def sbmv(self, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy):
        fn = getattr(self._api, 'cublas%shbmv_v2' % fmt)
        c_alpha = cty(alpha)
        c_beta = cty(beta)
        fn(self._handle, CUBLAS_FILL_MODE_MAP[uplo], n, k, byref(c_alpha),
           device_pointer(A), lda, device_pointer(x), incx, byref(c_beta),
           device_pointer(y), incy)

    return sbmv


def _Tspmv(fmt, cty, dtype):
    def sbmv(self, uplo, n, alpha, AP, x, incx, beta, y, incy):
        fn = getattr(self._api, 'cublas%sspmv_v2' % fmt)
        c_alpha = cty(alpha)
        c_beta = cty(beta)
        fn(self._handle, CUBLAS_FILL_MODE_MAP[uplo], n, byref(c_alpha),
           device_pointer(AP), device_pointer(x), incx, byref(c_beta),
           device_pointer(y), incy)

    return sbmv


def _Thpmv(fmt, cty, dtype):
    def sbmv(self, uplo, n, alpha, AP, x, incx, beta, y, incy):
        fn = getattr(self._api, 'cublas%shpmv_v2' % fmt)
        c_alpha = cty(alpha)
        c_beta = cty(beta)
        fn(self._handle, CUBLAS_FILL_MODE_MAP[uplo], n, byref(c_alpha),
           device_pointer(AP), device_pointer(x), incx, byref(c_beta),
           device_pointer(y), incy)

    return sbmv


def _Tger(fmt, cty, dtype):
    def ger(self, m, n, alpha, x, incx, y, incy, A, lda):
        fn = getattr(self._api, 'cublas%sger_v2' % fmt)
        c_alpha = cty(alpha)
        fn(self._handle, m, n, byref(c_alpha),
           device_pointer(x), incx, device_pointer(y), incy,
           device_pointer(A), lda)

    return ger


def _Tgeru(fmt, cty, dtype):
    def ger(self, m, n, alpha, x, incx, y, incy, A, lda):
        fn = getattr(self._api, 'cublas%sgeru_v2' % fmt)
        c_alpha = cty(alpha)
        fn(self._handle, m, n, byref(c_alpha),
           device_pointer(x), incx, device_pointer(y), incy,
           device_pointer(A), lda)

    return ger


def _Tgerc(fmt, cty, dtype):
    def ger(self, m, n, alpha, x, incx, y, incy, A, lda):
        fn = getattr(self._api, 'cublas%sgerc_v2' % fmt)
        c_alpha = cty(alpha)
        fn(self._handle, m, n, byref(c_alpha),
           device_pointer(x), incx, device_pointer(y), incy,
           device_pointer(A), lda)

    return ger


def _Tsyr(fmt, cty, dtype):
    def syr(self, uplo, n, alpha, x, incx, A, lda):
        fn = getattr(self._api, 'cublas%ssyr_v2' % fmt)
        c_alpha = cty(alpha)
        fn(self._handle, CUBLAS_FILL_MODE_MAP[uplo], n, byref(c_alpha),
           device_pointer(x), incx, device_pointer(A), lda)

    return syr


def _Ther(fmt, cty, dtype):
    def her(self, uplo, n, alpha, x, incx, A, lda):
        fn = getattr(self._api, 'cublas%sher_v2' % fmt)
        c_alpha = cty(alpha)
        fn(self._handle, CUBLAS_FILL_MODE_MAP[uplo], n, byref(c_alpha),
           device_pointer(x), incx, device_pointer(A), lda)

    return her


def _Tspr(fmt, cty, dtype):
    def spr(self, uplo, n, alpha, x, incx, AP):
        fn = getattr(self._api, 'cublas%sspr_v2' % fmt)
        c_alpha = cty(alpha)
        fn(self._handle, CUBLAS_FILL_MODE_MAP[uplo], n, byref(c_alpha),
           device_pointer(x), incx, device_pointer(AP))

    return spr


def _Thpr(fmt, cty, dtype):
    def hpr(self, uplo, n, alpha, x, incx, AP):
        fn = getattr(self._api, 'cublas%shpr_v2' % fmt)
        c_alpha = cty(alpha)
        fn(self._handle, CUBLAS_FILL_MODE_MAP[uplo], n, byref(c_alpha),
           device_pointer(x), incx, device_pointer(AP))

    return hpr


def _Tsyr2(fmt, cty, dtype):
    def syr2(self, uplo, n, alpha, x, incx, y, incy, A, lda):
        fn = getattr(self._api, 'cublas%ssyr2_v2' % fmt)
        c_alpha = cty(alpha)
        fn(self._handle, CUBLAS_FILL_MODE_MAP[uplo], n, byref(c_alpha),
           device_pointer(x), incx, device_pointer(y), incy, device_pointer(A),
           lda)

    return syr2


def _Ther2(fmt, cty, dtype):
    def her2(self, uplo, n, alpha, x, incx, y, incy, A, lda):
        fn = getattr(self._api, 'cublas%sher2_v2' % fmt)
        c_alpha = cty(alpha)
        fn(self._handle, CUBLAS_FILL_MODE_MAP[uplo], n, byref(c_alpha),
           device_pointer(x), incx, device_pointer(y), incy, device_pointer(A),
           lda)

    return her2


def _Tspr2(fmt, cty, dtype):
    def spr2(self, uplo, n, alpha, x, incx, y, incy, A):
        fn = getattr(self._api, 'cublas%sspr2_v2' % fmt)
        c_alpha = cty(alpha)
        fn(self._handle, CUBLAS_FILL_MODE_MAP[uplo], n, byref(c_alpha),
           device_pointer(x), incx, device_pointer(y), incy, device_pointer(A))

    return spr2


def _Thpr2(fmt, cty, dtype):
    def spr2(self, uplo, n, alpha, x, incx, y, incy, A):
        fn = getattr(self._api, 'cublas%shpr2_v2' % fmt)
        c_alpha = cty(alpha)
        fn(self._handle, CUBLAS_FILL_MODE_MAP[uplo], n, byref(c_alpha),
           device_pointer(x), incx, device_pointer(y), incy, device_pointer(A))

    return spr2


def _Tgemm(fmt, cty, dtype):
    def gemm(self, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C,
             ldc):
        fn = getattr(self._api, 'cublas%sgemm_v2' % fmt)
        c_alpha = cty(alpha)
        c_beta = cty(beta)
        fn(self._handle, CUBLAS_OP_MAP[transa], CUBLAS_OP_MAP[transb], m, n, k,
           byref(c_alpha), device_pointer(A), lda, device_pointer(B), ldb,
           byref(c_beta), device_pointer(C), ldc)

    return gemm


def _Tsyrk(fmt, cty, dtype):
    def syrk(self, uplo, trans, n, k, alpha, A, lda, beta, C, ldc):
        fn = getattr(self._api, 'cublas%ssyrk_v2' % fmt)
        c_alpha = cty(alpha)
        c_beta = cty(beta)
        fn(self._handle, CUBLAS_FILL_MODE_MAP[uplo], CUBLAS_OP_MAP[trans], n, k,
           byref(c_alpha), device_pointer(A), lda, byref(c_beta),
           device_pointer(C), ldc)

    return syrk


def _Therk(fmt, cty, dtype):
    def herk(self, uplo, trans, n, k, alpha, A, lda, beta, C, ldc):
        fn = getattr(self._api, 'cublas%sherk_v2' % fmt)
        c_alpha = cty(alpha)
        c_beta = cty(beta)
        fn(self._handle, CUBLAS_FILL_MODE_MAP[uplo], CUBLAS_OP_MAP[trans], n, k,
           byref(c_alpha), device_pointer(A), lda, byref(c_beta),
           device_pointer(C), ldc)

    return herk


def _Tsymm(fmt, cty, dtype):
    def symm(self, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc):
        fn = getattr(self._api, 'cublas%ssymm_v2' % fmt)
        c_alpha = cty(alpha)
        c_beta = cty(beta)
        fn(self._handle, CUBLAS_SIDE_MAP[side], CUBLAS_FILL_MODE_MAP[uplo], m,
           n,
           byref(c_alpha), device_pointer(A), lda, device_pointer(B), ldb,
           byref(c_beta), device_pointer(C), ldc)

    return symm


def _Themm(fmt, cty, dtype):
    def hemm(self, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc):
        fn = getattr(self._api, 'cublas%shemm_v2' % fmt)
        c_alpha = cty(alpha)
        c_beta = cty(beta)
        fn(self._handle, CUBLAS_SIDE_MAP[side], CUBLAS_FILL_MODE_MAP[uplo], m,
           n,
           byref(c_alpha), device_pointer(A), lda, device_pointer(B), ldb,
           byref(c_beta), device_pointer(C), ldc)

    return hemm


def _Ttrsm(fmt, cty, dtype):
    def trsm(self, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb):
        fn = getattr(self._api, 'cublas%strsm_v2' % fmt)
        c_alpha = cty(alpha)
        fn(self._handle, CUBLAS_SIDE_MAP[side], CUBLAS_FILL_MODE_MAP[uplo],
           CUBLAS_OP_MAP[trans], CUBLAS_DIAG_MAP[diag], m, n,
           byref(c_alpha), device_pointer(A), lda, device_pointer(B), ldb)

    return trsm


def _Ttrmm(fmt, cty, dtype):
    def trmm(self, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb,
             C, ldc):
        fn = getattr(self._api, 'cublas%strmm_v2' % fmt)
        c_alpha = cty(alpha)
        fn(self._handle, CUBLAS_SIDE_MAP[side], CUBLAS_FILL_MODE_MAP[uplo],
           CUBLAS_OP_MAP[trans], CUBLAS_DIAG_MAP[diag], m, n,
           byref(c_alpha), device_pointer(A), lda, device_pointer(B), ldb,
           device_pointer(C), ldc)

    return trmm


def _Tdgmm(fmt, cty, dtype):
    def dgmm(self, side, m, n, A, lda, x, incx, C, ldc):
        fn = getattr(self._api, 'cublas%sdgmm' % fmt)
        fn(self._handle, CUBLAS_SIDE_MAP[side], m, n, device_pointer(A), lda,
           device_pointer(x), incx, device_pointer(C), ldc)

    return dgmm


def _Tgeam(fmt, cty, dtype):
    def geam(self, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc):
        fn = getattr(self._api, 'cublas%sgeam' % fmt)
        c_alpha = cty(alpha)
        c_beta = cty(beta)
        fn(self._handle, CUBLAS_OP_MAP[transa], CUBLAS_OP_MAP[transb], m, n,
           byref(c_alpha), device_pointer(A), lda, byref(c_beta),
           device_pointer(B), ldb, device_pointer(C), ldc)

    return geam


class cuBlas(finalizer.OwnerMixin):
    def __init__(self):
        self._api = libcublas()
        self._handle = cublasHandle_t()
        self._api.cublasCreate_v2(byref(self._handle))
        self._finalizer_track((self._handle, self._api))
        self._stream = 0

    @classmethod
    def _finalize(self, res):
        handle, api = res
        api.cublasDestroy_v2(handle)

    @property
    def version(self):
        ver = c_int()
        self._api.cublasGetVersion_v2(self._handle, byref(ver))
        return ver.value

    @property
    def stream(self):
        return self._stream

    @stream.setter
    def stream(self, stream):
        self._stream = stream
        self._api.cublasSetStream_v2(self._handle, self.stream.handle)

    @property
    def pointer_mode(self):
        mode = cublasPointerMode_t()
        self._api.cublasGetPointerMode_v2(self._handle, byref(mode))
        return mode.value

    @pointer_mode.setter
    def pointer_mode(self, mode):
        self._api.cublasSetPointerMode_v2(self._handle, int(mode))

    @property
    def atomics_mode(self):
        mode = cublasAtomicsMode_t()
        self._api.cublasGetAtomicsMode(self._handle, byref(mode))
        return mode.value

    @atomics_mode.setter
    def atomics_mode(self, mode):
        self._api.cublasSetAtomicsMode(self._handle, int(mode))

    # Level 1

    Snrm2 = _Tnrm2('S', c_float)
    Dnrm2 = _Tnrm2('D', c_double)
    Scnrm2 = _Tnrm2('Sc', c_float)
    Dznrm2 = _Tnrm2('Dz', c_double)

    Sdot = _Tdot('S', c_float)
    Ddot = _Tdot('D', c_double)
    Cdotu = _Tdot('C', c_complex, 'u')
    Zdotu = _Tdot('Z', c_double_complex, 'u')
    Cdotc = _Tdot('C', c_complex, 'c')
    Zdotc = _Tdot('Z', c_double_complex, 'c')

    Sscal = _Tscal('S', c_float)
    Dscal = _Tscal('D', c_double)
    Cscal = _Tscal('C', c_complex)
    Zscal = _Tscal('Z', c_double_complex)
    Csscal = _Tscal('Cs', c_float)
    Zdscal = _Tscal('Zd', c_double)

    Saxpy = _Taxpy('S', c_float)
    Daxpy = _Taxpy('D', c_double)
    Caxpy = _Taxpy('C', c_complex)
    Zaxpy = _Taxpy('Z', c_double_complex)

    Isamax = _Itamax('s', c_float)
    Idamax = _Itamax('d', c_double)
    Icamax = _Itamax('c', c_complex)
    Izamax = _Itamax('z', c_double_complex)

    Isamin = _Itamin('s', c_float)
    Idamin = _Itamin('d', c_double)
    Icamin = _Itamin('c', c_complex)
    Izamin = _Itamin('z', c_double_complex)

    Sasum = _Tasum('S', c_float)
    Dasum = _Tasum('D', c_double)
    Scasum = _Tasum('Sc', c_float)
    Dzasum = _Tasum('Dz', c_double)

    Srot = _Trot('S', c_float, c_float)
    Drot = _Trot('D', c_double, c_double)
    Crot = _Trot('C', c_float, c_complex)
    Zrot = _Trot('Z', c_double, c_double_complex)
    Csrot = _Trot('Cs', c_float, c_float)
    Zdrot = _Trot('Zd', c_double, c_double)

    Srotg = _Trotg('S', c_float, c_float)
    Drotg = _Trotg('D', c_double, c_double)
    Crotg = _Trotg('C', c_complex, c_float)
    Zrotg = _Trotg('Z', c_double_complex, c_double)

    Srotm = _Trotm('S', np.float32)
    Drotm = _Trotm('D', np.float64)

    Srotmg = _Trotmg('S', c_float, np.float32)
    Drotmg = _Trotmg('D', c_double, np.float64)

    Sgbmv = _Tgbmv('S', c_float, np.float32)
    Dgbmv = _Tgbmv('D', c_double, np.float64)
    Cgbmv = _Tgbmv('C', c_complex, np.complex64)
    Zgbmv = _Tgbmv('Z', c_double_complex, np.complex128)

    Sgemv = _Tgemv('S', c_float, np.float32)
    Dgemv = _Tgemv('D', c_double, np.float64)
    Cgemv = _Tgemv('C', c_complex, np.complex64)
    Zgemv = _Tgemv('Z', c_double_complex, np.complex128)

    Strmv = _Ttrmv('S', np.float32)
    Dtrmv = _Ttrmv('D', np.float64)
    Ctrmv = _Ttrmv('C', np.complex64)
    Ztrmv = _Ttrmv('Z', np.complex128)

    Stbmv = _Ttbmv('S', np.float32)
    Dtbmv = _Ttbmv('D', np.float64)
    Ctbmv = _Ttbmv('C', np.complex64)
    Ztbmv = _Ttbmv('Z', np.complex128)

    Stpmv = _Ttpmv('S', np.float32)
    Dtpmv = _Ttpmv('D', np.float64)
    Ctpmv = _Ttpmv('C', np.complex64)
    Ztpmv = _Ttpmv('Z', np.complex128)

    Strsv = _Ttrsv('S', np.float32)
    Dtrsv = _Ttrsv('D', np.float64)
    Ctrsv = _Ttrsv('C', np.complex64)
    Ztrsv = _Ttrsv('Z', np.complex128)

    Stpsv = _Ttpsv('S', np.float32)
    Dtpsv = _Ttpsv('D', np.float64)
    Ctpsv = _Ttpsv('C', np.complex64)
    Ztpsv = _Ttpsv('Z', np.complex128)

    Stbsv = _Ttbsv('S', np.float32)
    Dtbsv = _Ttbsv('D', np.float64)
    Ctbsv = _Ttbsv('C', np.complex64)
    Ztbsv = _Ttbsv('Z', np.complex128)

    Ssymv = _Tsymv('S', c_float, np.float32)
    Dsymv = _Tsymv('D', c_double, np.float64)
    Csymv = _Tsymv('C', c_complex, np.complex64)
    Zsymv = _Tsymv('Z', c_double_complex, np.complex128)

    Chemv = _Themv('C', c_complex, np.complex64)
    Zhemv = _Themv('Z', c_double_complex, np.complex128)

    Ssbmv = _Tsbmv('S', c_float, np.float32)
    Dsbmv = _Tsbmv('D', c_double, np.float64)

    Chbmv = _Thbmv('C', c_complex, np.complex64)
    Zhbmv = _Thbmv('Z', c_double_complex, np.complex128)

    Sspmv = _Tspmv('S', c_float, np.float32)
    Dspmv = _Tspmv('D', c_double, np.float64)

    Chpmv = _Thpmv('C', c_complex, np.complex64)
    Zhpmv = _Thpmv('Z', c_double_complex, np.complex128)

    Sger = _Tger('S', c_float, np.float32)
    Dger = _Tger('D', c_double, np.float64)
    Cgeru = _Tgeru('C', c_complex, np.complex64)
    Cgerc = _Tgerc('C', c_complex, np.complex64)
    Zgeru = _Tgeru('Z', c_double_complex, np.complex128)
    Zgerc = _Tgerc('Z', c_double_complex, np.complex128)

    Ssyr = _Tsyr('S', c_float, np.float32)
    Dsyr = _Tsyr('D', c_double, np.float64)
    Csyr = _Tsyr('C', c_complex, np.complex64)
    Zsyr = _Tsyr('Z', c_double_complex, np.complex128)

    Cher = _Ther('C', c_float, np.complex64)
    Zher = _Ther('Z', c_double, np.complex128)

    Sspr = _Tspr('S', c_float, np.float32)
    Dspr = _Tspr('D', c_double, np.float64)
    Chpr = _Thpr('C', c_float, np.complex64)
    Zhpr = _Thpr('Z', c_double, np.complex128)

    Ssyr2 = _Tsyr2('S', c_float, np.float32)
    Dsyr2 = _Tsyr2('D', c_double, np.float64)
    Csyr2 = _Tsyr2('C', c_complex, np.complex64)
    Zsyr2 = _Tsyr2('Z', c_double_complex, np.complex128)
    Cher2 = _Ther2('C', c_complex, np.complex64)
    Zher2 = _Ther2('Z', c_double_complex, np.complex128)

    Sspr2 = _Tspr2('S', c_float, np.float32)
    Dspr2 = _Tspr2('D', c_double, np.float64)

    Chpr2 = _Thpr2('C', c_complex, np.complex64)
    Zhpr2 = _Thpr2('Z', c_double_complex, np.complex128)

    Sgemm = _Tgemm('S', c_float, np.float32)
    Dgemm = _Tgemm('D', c_double, np.float64)
    Cgemm = _Tgemm('C', c_complex, np.complex64)
    Zgemm = _Tgemm('Z', c_double_complex, np.complex128)

    Ssyrk = _Tsyrk('S', c_float, np.float32)
    Dsyrk = _Tsyrk('D', c_double, np.float64)
    Csyrk = _Tsyrk('C', c_complex, np.complex64)
    Zsyrk = _Tsyrk('Z', c_double_complex, np.complex128)

    Cherk = _Therk('C', c_float, np.complex64)
    Zherk = _Therk('Z', c_double, np.complex128)

    Ssymm = _Tsymm('S', c_float, np.float32)
    Dsymm = _Tsymm('D', c_double, np.float64)
    Csymm = _Tsymm('C', c_complex, np.complex64)
    Zsymm = _Tsymm('Z', c_double_complex, np.complex128)

    Chemm = _Themm('C', c_complex, np.complex64)
    Zhemm = _Themm('Z', c_double_complex, np.complex128)

    Strsm = _Ttrsm('S', c_float, np.float32)
    Dtrsm = _Ttrsm('D', c_double, np.float64)
    Ctrsm = _Ttrsm('C', c_complex, np.complex64)
    Ztrsm = _Ttrsm('Z', c_double_complex, np.complex128)

    Strmm = _Ttrmm('S', c_float, np.float32)
    Dtrmm = _Ttrmm('D', c_double, np.float64)
    Ctrmm = _Ttrmm('C', c_complex, np.complex64)
    Ztrmm = _Ttrmm('Z', c_double_complex, np.complex128)

    Sdgmm = _Tdgmm('S', c_float, np.float32)
    Ddgmm = _Tdgmm('D', c_double, np.float64)
    Cdgmm = _Tdgmm('C', c_complex, np.complex64)
    Zdgmm = _Tdgmm('Z', c_double_complex, np.complex128)

    Sgeam = _Tgeam('S', c_float, np.float32)
    Dgeam = _Tgeam('D', c_double, np.float64)
    Cgeam = _Tgeam('C', c_complex, np.complex64)
    Zgeam = _Tgeam('Z', c_double_complex, np.complex128)
