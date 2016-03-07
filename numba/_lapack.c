/*
 * This file contains wrappers of BLAS and LAPACK functions
 */
#include <complex.h>

/*
 * BLAS calling helpers.  The helpers can be called without the GIL held.
 * The caller is responsible for checking arguments (especially dimensions).
 */

/* Fast getters caching the value of a function's address after
   the first call to import_cblas_function(). */

#define EMIT_GET_CBLAS_FUNC(name)                                 \
    static void *cblas_ ## name = NULL;                           \
    static void *get_cblas_ ## name(void) {                       \
        if (cblas_ ## name == NULL) {                             \
            PyGILState_STATE st = PyGILState_Ensure();            \
            const char *mod = "scipy.linalg.cython_blas";         \
            cblas_ ## name = import_cython_function(mod, # name); \
            PyGILState_Release(st);                               \
        }                                                         \
        return cblas_ ## name;                                    \
    }

EMIT_GET_CBLAS_FUNC(dgemm)
EMIT_GET_CBLAS_FUNC(sgemm)
EMIT_GET_CBLAS_FUNC(cgemm)
EMIT_GET_CBLAS_FUNC(zgemm)
EMIT_GET_CBLAS_FUNC(dgemv)
EMIT_GET_CBLAS_FUNC(sgemv)
EMIT_GET_CBLAS_FUNC(cgemv)
EMIT_GET_CBLAS_FUNC(zgemv)
EMIT_GET_CBLAS_FUNC(ddot)
EMIT_GET_CBLAS_FUNC(sdot)
EMIT_GET_CBLAS_FUNC(cdotu)
EMIT_GET_CBLAS_FUNC(zdotu)
EMIT_GET_CBLAS_FUNC(cdotc)
EMIT_GET_CBLAS_FUNC(zdotc)

#undef EMIT_GET_CBLAS_FUNC

/*
 * Define what a Fortran "int" is, some LAPACKs have 64 bit integer support
 * numba presently opts for a 32 bit C int.
 * This definition allows scope for later configuration time magic to adjust
 * the size of int at all the call sites.
 */
#define F_INT int

typedef float (*sdot_t)(F_INT *n, void *dx, F_INT *incx, void *dy, F_INT *incy);
typedef double (*ddot_t)(F_INT *n, void *dx, F_INT *incx, void *dy, F_INT 
*incy);
typedef npy_complex64 (*cdot_t)(F_INT *n, void *dx, F_INT *incx, void *dy, 
F_INT *incy);
typedef npy_complex128 (*zdot_t)(F_INT *n, void *dx, F_INT *incx, void *dy, 
F_INT *incy);

typedef void (*xxgemv_t)(char *trans, F_INT *m, F_INT *n,
                         void *alpha, void *a, F_INT *lda,
                         void *x, F_INT *incx, void *beta,
                         void *y, F_INT *incy);

typedef void (*xxgemm_t)(char *transa, char *transb,
                         F_INT *m, F_INT *n, F_INT *k,
                         void *alpha, void *a, F_INT *lda,
                         void *b, F_INT *ldb, void *beta,
                         void *c, F_INT *ldc);

/* Vector * vector: result = dx * dy */
NUMBA_EXPORT_FUNC(F_INT)
numba_xxdot(char kind, char conjugate, F_INT n, void *dx, void *dy,
            void *result)
{
    void *raw_func = NULL;
    F_INT _n;
    F_INT inc = 1;

    switch (kind) {
        case 'd':
            raw_func = get_cblas_ddot();
            break;
        case 's':
            raw_func = get_cblas_sdot();
            break;
        case 'c':
            raw_func = conjugate ? get_cblas_cdotc() : get_cblas_cdotu();
            break;
        case 'z':
            raw_func = conjugate ? get_cblas_zdotc() : get_cblas_zdotu();
            break;
        default:
            {
                PyGILState_STATE st = PyGILState_Ensure();
                PyErr_SetString(PyExc_ValueError,
                                "invalid kind of *DOT function");
                PyGILState_Release(st);
            }
            return -1;
    }
    if (raw_func == NULL)
        return -1;

    _n = (F_INT) n;

    switch (kind) {
        case 'd':
            *(double *) result = (*(ddot_t) raw_func)(&_n, dx, &inc, dy, &inc);;
            break;
        case 's':
            *(float *) result = (*(sdot_t) raw_func)(&_n, dx, &inc, dy, &inc);;
            break;
        case 'c':
            *(npy_complex64 *) result = (*(cdot_t) raw_func)(&_n, dx, &inc, dy,\
                    &inc);;
            break;
        case 'z':
            *(npy_complex128 *) result = (*(zdot_t) raw_func)(&_n, dx, &inc,\
                    dy, &inc);;
            break;
    }

    return 0;
}

/* Matrix * vector: y = alpha * a * x + beta * y */
NUMBA_EXPORT_FUNC(F_INT)
numba_xxgemv(char kind, char *trans, F_INT m, F_INT n,
             void *alpha, void *a, F_INT lda,
             void *x, void *beta, void *y)
{
    void *raw_func = NULL;
    F_INT _m, _n;
    F_INT _lda;
    F_INT inc = 1;

    switch (kind) {
        case 'd':
            raw_func = get_cblas_dgemv();
            break;
        case 's':
            raw_func = get_cblas_sgemv();
            break;
        case 'c':
            raw_func = get_cblas_cgemv();
            break;
        case 'z':
            raw_func = get_cblas_zgemv();
            break;
        default:
            {
                PyGILState_STATE st = PyGILState_Ensure();
                PyErr_SetString(PyExc_ValueError,
                                "invalid kind of *GEMV function");
                PyGILState_Release(st);
            }
            return -1;
    }
    if (raw_func == NULL)
        return -1;

    _m = (F_INT) m;
    _n = (F_INT) n;
    _lda = (F_INT) lda;

    (*(xxgemv_t) raw_func)(trans, &_m, &_n, alpha, a, &_lda,
                           x, &inc, beta, y, &inc);
    return 0;
}

/* Matrix * matrix: c = alpha * a * b + beta * c */
NUMBA_EXPORT_FUNC(F_INT)
numba_xxgemm(char kind, char *transa, char *transb,
             F_INT m, F_INT n, F_INT k,
             void *alpha, void *a, F_INT lda,
             void *b, F_INT ldb, void *beta,
             void *c, F_INT ldc)
{
    void *raw_func = NULL;
    F_INT _m, _n, _k;
    F_INT _lda, _ldb, _ldc;

    switch (kind) {
        case 'd':
            raw_func = get_cblas_dgemm();
            break;
        case 's':
            raw_func = get_cblas_sgemm();
            break;
        case 'c':
            raw_func = get_cblas_cgemm();
            break;
        case 'z':
            raw_func = get_cblas_zgemm();
            break;
        default:
            {
                PyGILState_STATE st = PyGILState_Ensure();
                PyErr_SetString(PyExc_ValueError,
                                "invalid kind of *GEMM function");
                PyGILState_Release(st);
            }
            return -1;
    }
    if (raw_func == NULL)
        return -1;

    _m = (F_INT) m;
    _n = (F_INT) n;
    _k = (F_INT) k;
    _lda = (F_INT) lda;
    _ldb = (F_INT) ldb;
    _ldc = (F_INT) ldc;

    (*(xxgemm_t) raw_func)(transa, transb, &_m, &_n, &_k, alpha, a, &_lda,
                           b, &_ldb, beta, c, &_ldc);
    return 0;
}

/*
 * LAPACK calling helpers.  The helpers can be called without the GIL held.
 * The caller is responsible for checking arguments (especially dimensions).
 */

/* Fast getters caching the value of a function's address after
   the first call to import_clapack_function(). */

#define EMIT_GET_CLAPACK_FUNC(name)                                 \
    static void *clapack_ ## name = NULL;                           \
    static void *get_clapack_ ## name(void) {                       \
        if (clapack_ ## name == NULL) {                             \
            PyGILState_STATE st = PyGILState_Ensure();              \
            const char *mod = "scipy.linalg.cython_lapack";         \
            clapack_ ## name = import_cython_function(mod, # name); \
            PyGILState_Release(st);                                 \
        }                                                           \
        return clapack_ ## name;                                    \
    }

// Computes an LU factorization of a general M-by-N matrix A
// using partial pivoting with row interchanges.
EMIT_GET_CLAPACK_FUNC(sgetrf)
EMIT_GET_CLAPACK_FUNC(dgetrf)
EMIT_GET_CLAPACK_FUNC(cgetrf)
EMIT_GET_CLAPACK_FUNC(zgetrf)

// Computes the inverse of a matrix using the LU factorization
// computed by xGETRF.
EMIT_GET_CLAPACK_FUNC(sgetri)
EMIT_GET_CLAPACK_FUNC(dgetri)
EMIT_GET_CLAPACK_FUNC(cgetri)
EMIT_GET_CLAPACK_FUNC(zgetri)

// Compute Cholesky factorizations
EMIT_GET_CLAPACK_FUNC(spotrf)
EMIT_GET_CLAPACK_FUNC(dpotrf)
EMIT_GET_CLAPACK_FUNC(cpotrf)
EMIT_GET_CLAPACK_FUNC(zpotrf)

// Computes for an N-by-N real nonsymmetric matrix A, the
// eigenvalues and, optionally, the left and/or right eigenvectors.
EMIT_GET_CLAPACK_FUNC(sgeev)
EMIT_GET_CLAPACK_FUNC(dgeev)
EMIT_GET_CLAPACK_FUNC(cgeev)
EMIT_GET_CLAPACK_FUNC(zgeev)

// Computes generalised SVD
EMIT_GET_CLAPACK_FUNC(sgesdd)
EMIT_GET_CLAPACK_FUNC(dgesdd)
EMIT_GET_CLAPACK_FUNC(cgesdd)
EMIT_GET_CLAPACK_FUNC(zgesdd)


#undef EMIT_GET_CLAPACK_FUNC

typedef void (*xxgetrf_t)(F_INT *m, F_INT *n, void *a, F_INT *lda, F_INT *ipiv, 
F_INT *info);

typedef void (*xxgetri_t)(F_INT *n, void *a, F_INT *lda, F_INT *ipiv, void 
*work, F_INT *lwork, F_INT *info);

typedef void (*xxpotrf_t)(char *uplo, F_INT *n, void *a, F_INT *lda, F_INT 
*info);

typedef void (*rgeev_t)(char *jobvl, char *jobvr, F_INT *n, void *a, F_INT *lda,
void *wr, void *wi, void *vl, F_INT *ldvl, void *vr, F_INT *ldvr, void *work,
F_INT *lwork, F_INT *info);

typedef void (*cgeev_t)(char *jobvl, char *jobvr, F_INT *n, void *a, F_INT 
*lda, void *w, void *vl, F_INT *ldvl, void *vr, F_INT *ldvr, void *work, F_INT 
*lwork, double *rwork, F_INT *info);

typedef void (*rgesdd_t)(char *jobz, F_INT *m, F_INT *n, void *a, F_INT *lda, 
                         void *s, void *u, F_INT *ldu, void *vt, F_INT *ldvt,   
                         void *work, F_INT *lwork, F_INT *iwork, F_INT *info);

typedef void (*cgesdd_t)(char *jobz, F_INT *m, F_INT *n, void *a, F_INT *lda, 
                         void *s, void * u, F_INT *ldu, void * vt, F_INT *ldvt, 
                         void *work, F_INT *lwork, void *rwork, F_INT *iwork,
                         F_INT *info);

/* Compute LU decomposition of A */
NUMBA_EXPORT_FUNC(F_INT)
numba_xxgetrf(char kind, F_INT m, F_INT n, void *a, F_INT lda, 
F_INT *ipiv, F_INT *info)
{
    void *raw_func = NULL;
    F_INT _m, _n, _lda;

    switch (kind) {
        case 's':
            raw_func = get_clapack_sgetrf();
            break;
        case 'd':
            raw_func = get_clapack_dgetrf();
            break;
        case 'c':
            raw_func = get_clapack_cgetrf();
            break;
        case 'z':
            raw_func = get_clapack_zgetrf();
            break;
        default:
            {
                PyGILState_STATE st = PyGILState_Ensure();
                PyErr_SetString(PyExc_ValueError,
                                "invalid kind of *LU factorization function");
                PyGILState_Release(st);
            }
            return -1;
    }
    if (raw_func == NULL)
        return -1;

    _m = (F_INT) m;
    _n = (F_INT) n;
    _lda = (F_INT) lda;

    (*(xxgetrf_t) raw_func)(&_m, &_n, a, &_lda, ipiv, info);
    return 0;
}


/* Compute the inverse of a matrix given its LU decomposition*/
NUMBA_EXPORT_FUNC(F_INT)
numba_xxgetri(char kind, F_INT n, void *a, F_INT lda, F_INT *ipiv, \
    void *work, F_INT *lwork, F_INT *info)
{
    void *raw_func = NULL;
    F_INT _n, _lda;

    switch (kind) {
        case 's':
            raw_func = get_clapack_sgetri();
            break;
        case 'd':
            raw_func = get_clapack_dgetri();
            break;
        case 'c':
            raw_func = get_clapack_cgetri();
            break;
        case 'z':
            raw_func = get_clapack_zgetri();
            break;
        default:
            {
                PyGILState_STATE st = PyGILState_Ensure();
                PyErr_SetString(PyExc_ValueError,\
                  "invalid kind of *inversion from LU factorization function");
                PyGILState_Release(st);
            }
            return -1;
    }
    if (raw_func == NULL)
        return -1;

    _n = (F_INT) n;
    _lda = (F_INT) lda;

    (*(xxgetri_t) raw_func)(&_n, a, &_lda, ipiv, work, lwork, info);
    return 0;
}

/* Compute the Cholesky factorization of a matrix */
NUMBA_EXPORT_FUNC(int)
numba_xxpotrf(char kind, char uplo, Py_ssize_t n, void *a, Py_ssize_t lda, int 
*info)
{
    void *raw_func = NULL;
    int _n, _lda;

    switch (kind) {
        case 's':
            raw_func = get_clapack_spotrf();
            break;
        case 'd':
            raw_func = get_clapack_dpotrf();
            break;
        case 'c':
            raw_func = get_clapack_cpotrf();
            break;
        case 'z':
            raw_func = get_clapack_zpotrf();
            break;
        default:
            {
                PyGILState_STATE st = PyGILState_Ensure();
                PyErr_SetString(PyExc_ValueError,
                            "invalid kind of Cholesky factorization function");
                PyGILState_Release(st);
            }
            return -1;
    }
    if (raw_func == NULL)
        return -1;

    _n = (int) n;
    _lda = (int) lda;

    (*(xxpotrf_t) raw_func)(&uplo, &_n, a, &_lda, info);
    return 0;
}


/*
 * cast_from_X()
 * cast from a kind (s, d, c, z) = (float, double, complex, double complex)
 * to a Fortran integer.
 *
 * Parameters:
 * kind the kind of val
 * val  a pointer to the value to cast
 *
 * Returns: 
 * A Fortran int from a cast of val (in complex case, takes the real part).
 *
 */
// a template would be nice
// struct access via non c99 (python only) cmplx types, used for compatibility
F_INT cast_from_X(char kind, void *val)
{
    switch(kind) {
        case 's':
            return (F_INT)(*((float *) val));
        case 'd':
            return (F_INT)(*((double *) val));
        case 'c':
            return (F_INT)(*((npy_complex64 *)val)).real;
        case 'z':
            return (F_INT)(*((npy_complex128 *)val)).real;
        default:
            {
                PyGILState_STATE st = PyGILState_Ensure();
                PyErr_SetString(PyExc_ValueError,\
                  "invalid kind in cast");
                PyGILState_Release(st);
            }
    }
    return -1;
}

// real space eigen systems info from dgeev/sgeev
NUMBA_EXPORT_FUNC(F_INT)
numba_raw_rgeev(char kind, char jobvl, char jobvr, 
F_INT n, void *a, F_INT lda, void *wr, void *wi, void *vl, 
F_INT ldvl, void *vr, F_INT ldvr, void *work, F_INT lwork, F_INT * info)
{
    void *raw_func = NULL;

    switch (kind) {
        case 's':
            raw_func = get_clapack_sgeev();
            break;
        case 'd':
            raw_func = get_clapack_dgeev();
            break;
        default:
            {
                PyGILState_STATE st = PyGILState_Ensure();
                PyErr_SetString(PyExc_ValueError,\
                  "invalid kind of real space *geev call");
                PyGILState_Release(st);
            }
            return -1;
    }
    if (raw_func == NULL)
        return -1;

    (*(rgeev_t) raw_func)(&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr,
        &ldvr, work, &lwork, info);
    return 0;
}

// real space eigen systems info from dgeev/sgeev
// as numba_raw_rgeev but the allocation and error handling is done for the user
// Args are as per LAPACK.
NUMBA_EXPORT_FUNC(F_INT)
numba_ez_rgeev(char kind, char jobvl, char jobvr, 
F_INT n, void *a, F_INT lda, void *wr, void *wi, void *vl, 
F_INT ldvl, void *vr, F_INT ldvr)
{
    F_INT info = 0;
    size_t base_size = -1;
    // find the function to call, decide on a base type size
    switch (kind) {
        case 's':
            base_size = sizeof(float);
            break;
        case 'd':
            base_size = sizeof(double);
            break;
        default:
            {
                PyGILState_STATE st = PyGILState_Ensure();
                PyErr_SetString(PyExc_ValueError,\
                  "Invalid kind in numba_ez_rgeev");
                PyGILState_Release(st);
            }
            return -1;
    } 
    F_INT lwork = -1;
    void * work = malloc(base_size);
    numba_raw_rgeev(kind, jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, 
                vr, ldvr, work, lwork, &info);
    lwork = cast_from_X(kind, work);
    free(work);
    if(info < 0) {
        {
            PyGILState_STATE st = PyGILState_Ensure();
            PyErr_Format(PyExc_ValueError,\
                "LAPACK Error: on input %d\n", -info);
            PyGILState_Release(st);
        }
        return -1;
    }
    work = malloc(base_size * lwork);
    numba_raw_rgeev(kind, jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, 
                vr, ldvr, work, lwork, &info);
    free(work);

    if(info) {
        {
            PyGILState_STATE st = PyGILState_Ensure();
            if(info < 0) {
                PyErr_Format(PyExc_ValueError,\
                "LAPACK Error: on input %d\n", -info);
            } else 
            {
            PyErr_Format(PyExc_ValueError,\
                "LAPACK Error: QR failed to compute all eigenvalues, no\
                eigenvectors have been computed. i+1:n of wr/wi contains\
                converged eigenvalues. i = %d (Fortran indexing)\n", info);
            }
            PyGILState_Release(st);
        }
        return -1;
    }
    return 0;
}

// complex space eigen systems info from cgeev/zgeev
// Args are as per LAPACK.
NUMBA_EXPORT_FUNC(F_INT)
numba_raw_cgeev(char kind, char jobvl, char jobvr,
F_INT n, void *a, F_INT lda, void *w, void *vl, F_INT ldvl, void
*vr, F_INT ldvr, void *work, F_INT lwork, double *rwork, F_INT * info)
{
    void *raw_func = NULL;

    switch (kind) {
        case 'c':
            raw_func = get_clapack_cgeev();
            break;
        case 'z':
            raw_func = get_clapack_zgeev();
            break;
        default:
            {
                PyGILState_STATE st = PyGILState_Ensure();
                PyErr_SetString(PyExc_ValueError,\
                  "invalid kind of complex space *geev call");
                PyGILState_Release(st);
            }
            return -1;
    }
    if (raw_func == NULL)
        return -1;

    (*(cgeev_t) raw_func)(&jobvl, &jobvr, &n, a, &lda, w, vl, &ldvl, vr, &ldvr,
        work, &lwork, rwork, info);
    return 0;
}


// complex space eigen systems info from cgeev/zgeev
// as numba_raw_cgeev but the allocation and error handling is done for the user
// Args are as per LAPACK.
NUMBA_EXPORT_FUNC(F_INT)
numba_ez_cgeev(char kind, char jobvl, char jobvr, 
F_INT n, void *a, F_INT lda, void *w, void *vl, F_INT ldvl, void *vr, F_INT 
ldvr)
{
    F_INT info = 0;
    size_t base_size = -1;
    // find the function to call, decide on a base type size
    switch (kind) {
        case 'c':
            base_size = sizeof(npy_complex64);
            break;
        case 'z':
            base_size = sizeof(npy_complex128);
            break;
        default:
            {
                PyGILState_STATE st = PyGILState_Ensure();
                PyErr_SetString(PyExc_ValueError,\
                  "Invalid kind in numba_ez_cgeev");
                PyGILState_Release(st);
            }
            return -1;
    }       
    F_INT lwork = -1;
    void * work = malloc(base_size);
    double * rwork = malloc(2*n*base_size);
    numba_raw_cgeev(kind, jobvl, jobvr, n, a, lda, w, vl, ldvl, 
                vr, ldvr, work, lwork, rwork, &info);
    lwork = cast_from_X(kind, work);
    free(work);
    if(info < 0) {
        free(rwork);
        {
            PyGILState_STATE st = PyGILState_Ensure();
            PyErr_Format(PyExc_ValueError,\
                "LAPACK Error: on input %d\n", -info);
            PyGILState_Release(st);
        }
        return -1;
    }
    work = malloc(base_size * lwork);
    numba_raw_cgeev(kind, jobvl, jobvr, n, a, lda, w, vl, ldvl, 
                vr, ldvr, work, lwork, rwork, &info);
    free(work);
    free(rwork);

    if(info) {
        {
            PyGILState_STATE st = PyGILState_Ensure();
            if(info < 0) {
                PyErr_Format(PyExc_ValueError,\
                "LAPACK Error: on input %d\n", -info);
            } else 
            {
            PyErr_Format(PyExc_ValueError,\
                "LAPACK Error: QR failed to compute all eigenvalues, no\
                eigenvectors have been computed. i+1:n of w contains\
                converged eigenvalues. i = %d (Fortran indexing)\n", info);
            }
            PyGILState_Release(st);
        }
        return -1;
    }
    return 0;
}



// Eigen systems info from *geev.
// This routine hides the type and general complexity involved with making the
// calls to *geev. The signatures for the real/complex space routines differ
// in that the real space routines return real and complex eigenvalues in
// separate real variables, this is hidden below by packing them into a complex
// variable. The work space computation and error handling etc is also hidden.
// Args are as per LAPACK.
NUMBA_EXPORT_FUNC(F_INT)
numba_ez_geev(char kind, char jobvl, char jobvr,
F_INT n, void *a, F_INT lda, void *w, void *vl, F_INT ldvl, void
*vr, F_INT ldvr)
{
    // real space, will need packing into `w` for return
    void * wr = NULL, * wi = NULL;
    size_t base_size;
    switch (kind) {
        case 's':
            base_size = sizeof(float);
        case 'd':
            base_size = sizeof(double);
            
            wi = malloc(n*base_size);
            wr = malloc(n*base_size);

            numba_ez_rgeev(kind, jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, 
                        vr, ldvr);
            break;
        case 'c':
        case 'z':
            numba_ez_cgeev(kind, jobvl, jobvr, n, a, lda, w, vl, ldvl, 
                        vr, ldvr);
            break;
        default:
            {
                PyGILState_STATE st = PyGILState_Ensure();
                PyErr_SetString(PyExc_ValueError,\
                  "invalid kind in numba_ez_geev call");
                PyGILState_Release(st);
            }
            return -1;
    }

    F_INT k;
    npy_complex64 * c64_ptr;
    npy_complex128 * c128_ptr;
    switch (kind) {
    case 's':
        c64_ptr = (npy_complex64 *)w;
        for(k = 0; k < n; k++)
        {
            c64_ptr[k].real = ((float *)wr)[k];
            c64_ptr[k].imag = ((float *)wi)[k];
        }
        break;
    case 'd':
        c128_ptr = (npy_complex128 *)w;
        for(k = 0; k < n; k++)
        {
            c128_ptr[k].real = ((double *)wr)[k];
            c128_ptr[k].imag = ((double *)wi)[k];
        }
        break;
    }
    return 0;
}

// real space svd systems info from dgesdd/sgesdd
// Args are as per LAPACK.
NUMBA_EXPORT_FUNC(F_INT)
numba_raw_rgesdd(char kind, char jobz, F_INT m, F_INT n, void *a, F_INT lda, 
                 void *s,  void *u, F_INT ldu, void *vt,  F_INT ldvt, 
                 void *work, F_INT lwork, F_INT *iwork, F_INT *info)
{
    void *raw_func = NULL;

    switch (kind) {
        case 's':
            raw_func = get_clapack_sgesdd();
            break;
        case 'd':
            raw_func = get_clapack_dgesdd();
            break;
        default:
            {
                PyGILState_STATE st = PyGILState_Ensure();
                PyErr_SetString(PyExc_ValueError,\
                  "invalid kind of real space *gesdd call");
                PyGILState_Release(st);
            }
            return -1;
    }
    if (raw_func == NULL)
        return -1;

    (*(rgesdd_t) raw_func)(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work,
                           &lwork, iwork, info);
    return 0;
}

// real space svd info from dgesdd/sgesdd.
// As numba_raw_rgesdd but the allocation and error handling is done for the 
// user
// Args are as per LAPACK.
NUMBA_EXPORT_FUNC(F_INT)
numba_ez_rgesdd(char kind, char jobz, F_INT m, F_INT n, void *a, F_INT lda, 
                void *s, void *u, F_INT ldu, void *vt,  F_INT ldvt)
{
    F_INT info = 0;
    size_t base_size = -1;
    // find the function to call, decide on a base type size
    switch (kind) {
        case 's':
            base_size = sizeof(float);
            break;
        case 'd':
            base_size = sizeof(double);
            break;
        default:
            {
                PyGILState_STATE st = PyGILState_Ensure();
                PyErr_SetString(PyExc_ValueError,\
                  "Invalid kind in numba_ez_rgesdd");
                PyGILState_Release(st);
            }
            return -1;
    }
      
    F_INT lwork = -1;
    void *work = malloc(base_size);
    void *iwork = NULL;
    numba_raw_rgesdd(kind, jobz, m, n, a, lda, s, u ,ldu, vt, ldvt, work, lwork,
                     iwork, &info);
    lwork = cast_from_X(kind, work);
    free(work);

    if(info < 0) {
        {
            PyGILState_STATE st = PyGILState_Ensure();
            PyErr_Format(PyExc_ValueError,\
                "LAPACK Error: on input %d\n", -info);
            PyGILState_Release(st);
        }
        return -1;
    }
    work = malloc(base_size * lwork);
    F_INT minmn = m > n ? n : m;
    iwork = malloc(8 * minmn * sizeof(F_INT));
    
    numba_raw_rgesdd(kind, jobz, m, n, a, lda, s, u ,ldu, vt, ldvt, work, lwork,
                     iwork, &info);

    free(work);
    free(iwork);

    if(info) {
        {
            PyGILState_STATE st = PyGILState_Ensure();
            if(info < 0) {
                PyErr_Format(PyExc_ValueError,\
                "LAPACK Error: on input %d\n", -info);
            } else 
            {
            PyErr_Format(PyExc_ValueError,\
                "LAPACK Error: Convergence of internal algorithm reported \
                failure. There were %d superdiagonal elements that failed to \
                converge.\n", info);
            }
            PyGILState_Release(st);
        }
        return -1;
    }
    return 0;
}


// complex space svd systems info from cgesdd/zgesdd
// Args are as per LAPACK.
NUMBA_EXPORT_FUNC(F_INT)
numba_raw_cgesdd(char kind, char jobz, F_INT m, F_INT n, void *a, F_INT lda,
                 void *s, void *u, F_INT ldu, void *vt, F_INT ldvt, void *work,
                 F_INT lwork, void *rwork, F_INT *iwork, F_INT *info)
{
    void *raw_func = NULL;

    switch (kind) {
        case 'c':
            raw_func = get_clapack_cgesdd();
            break;
        case 'z':
            raw_func = get_clapack_zgesdd();
            break;
        default:
            {
                PyGILState_STATE st = PyGILState_Ensure();
                PyErr_SetString(PyExc_ValueError,\
                  "invalid kind of complex space *gesdd call");
                PyGILState_Release(st);
            }
            return -1;
    }
    if (raw_func == NULL)
        return -1;

    (*(cgesdd_t) raw_func)(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work,
                           &lwork, rwork, iwork, info);
    return 0;
}

// complex space svd info from cgesdd/zgesdd.
// As numba_raw_cgesdd but the allocation and error handling is done for the 
// user
// Args are as per LAPACK.
NUMBA_EXPORT_FUNC(F_INT)
numba_ez_cgesdd(char kind, char jobz, F_INT m, F_INT n, void *a, F_INT lda, 
                void *s, void *u, F_INT ldu, void *vt,  F_INT ldvt)
{
    F_INT info = 0;
    size_t real_base_size = -1;
    size_t complex_base_size = -1;
    // find the function to call, decide on a base type size
    switch (kind) {
        case 'c':
            real_base_size = sizeof(float);
            complex_base_size = sizeof(npy_complex64);
            break;
        case 'z':
            real_base_size = sizeof(double);
            complex_base_size = sizeof(npy_complex128);
            break;
        default:
            {
                PyGILState_STATE st = PyGILState_Ensure();
                PyErr_SetString(PyExc_ValueError,\
                  "Invalid kind in numba_ez_rgesdd");
                PyGILState_Release(st);
            }
            return -1;
    }
      
    F_INT lwork = -1;
    void *work = malloc(complex_base_size);
    void *rwork = NULL;
    void *iwork = NULL;
    numba_raw_cgesdd(kind, jobz, m, n, a, lda, s, u ,ldu, vt, ldvt, work, lwork,
                     rwork, iwork, &info);
    lwork = cast_from_X(kind, work);
    free(work);
    if(info < 0) {
        {
            PyGILState_STATE st = PyGILState_Ensure();
            PyErr_Format(PyExc_ValueError,\
                "LAPACK Error: on input %d\n", -info);
            PyGILState_Release(st);
        }
        return -1;
    }
    work = malloc(complex_base_size * lwork);
    
    F_INT lrwork;
    F_INT minmn = m > n ? n : m;
    if (jobz == 'n')
    {
        lrwork = 7 * minmn;
    }
    else
    {
        F_INT maxmn = m > n ? m : n;
        F_INT tmp1 = 5 * minmn + 7;
        F_INT tmp2 = 2 * maxmn + 2 * minmn + 1;
        lrwork = minmn * (tmp1 > tmp2 ? tmp1: tmp2);
    }
    rwork = malloc(real_base_size * (lrwork > 1 ? lrwork : 1));
      
    numba_raw_cgesdd(kind, jobz, m, n, a, lda, s, u ,ldu, vt, ldvt, work, lwork,
                     rwork, iwork, &info);
    
    free(work);
    free(rwork);

    if(info) {
        {
            PyGILState_STATE st = PyGILState_Ensure();
            if(info < 0) {
                PyErr_Format(PyExc_ValueError,\
                "LAPACK Error: on input %d\n", -info);
            } else 
            {
            PyErr_Format(PyExc_ValueError,\
                "LAPACK Error: Convergence of internal algorithm reported \
                failure. There were %d superdiagonal elements that failed to \
                converge.\n", info);
            }
            PyGILState_Release(st);
        }
        return -1;
    }
    return 0;
}


// SVD systems info from *gesdd.
// This routine hides the type and general complexity involved with making the
// calls to *gesdd. The work space computation and error handling etc is hidden.
// Args are as per LAPACK.
NUMBA_EXPORT_FUNC(F_INT)
numba_ez_gesdd(char kind, char jobz, F_INT m, F_INT n, void *a, F_INT lda, 
                void *s, void *u, F_INT ldu, void *vt,  F_INT ldvt)
{
    switch (kind) {
        case 's':
        case 'd':
            numba_ez_rgesdd(kind, jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
            break;
        case 'c':
        case 'z':
            numba_ez_cgesdd(kind, jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
            break;
        default:
            {
                PyGILState_STATE st = PyGILState_Ensure();
                PyErr_SetString(PyExc_ValueError,\
                  "invalid kind in numba_ez_gesdd call");
                PyGILState_Release(st);
            }
            return -1;
    }
    return 0;
}
