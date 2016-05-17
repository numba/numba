/*
 * This file contains wrappers of BLAS and LAPACK functions
 */
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
 * A union of all the types accepted by BLAS/LAPACK for use in cases where
 * stack based allocation is needed (typically for work space query args length
 * 1).
 */
typedef union all_dtypes_
{
    float  s;
    double d;
    npy_complex64 c;
    npy_complex128 z;
} all_dtypes;

/*
 * A checked PyMem_RawMalloc, ensures that the var is either NULL
 * and an exception is raised, or that the allocation was successful.
 * Returns zero on success for status checking.
 */
static int checked_PyMem_RawMalloc(void** var, size_t bytes)
{
    *var = NULL;
    *var = PyMem_RawMalloc(bytes);
    if (!(*var))
    {
        {
            PyGILState_STATE st = PyGILState_Ensure();

            PyErr_SetString(PyExc_MemoryError,
                            "Insufficient memory for buffer allocation\
                             required by LAPACK.");
            PyGILState_Release(st);
        }
        return 1;
    }
    return 0;
}

/*
 * Checks that the char kind is valid (one of [s,d,c,z]) for use in blas/lapack.
 * Returns zero on success for status checking.
 */
static int check_kind(char kind)
{
    switch (kind)
    {
        case 's':
        case 'd':
        case 'c':
        case 'z':
            break;
        default:
        {
            PyGILState_STATE st = PyGILState_Ensure();
            PyErr_SetString(PyExc_ValueError,
                            "invalid data type (kind) found");
            PyGILState_Release(st);
        }
        return 1;
    }
    return 0;
}

/*
 * Guard macro for ensuring a valid data "kind" is being used.
 * Place at the top of all routines with switches on "kind" that accept
 * one of [s,d,c,z].
 */
#define ENSURE_VALID_KIND(__KIND) \
if (check_kind( __KIND ))         \
{                                 \
    return -1;                    \
}                                 \

/*
 * Checks that the char kind is valid for the real domain (one of [s,d])
 * for use in blas/lapack.
 * Returns zero on success for status checking.
 */
static int check_real_kind(char kind)
{
    switch (kind)
    {
        case 's':
        case 'd':
            break;
        default:
        {
            PyGILState_STATE st = PyGILState_Ensure();
            PyErr_SetString(PyExc_ValueError,
                            "invalid data type (kind) found");
            PyGILState_Release(st);
        }
        return 1;
    }
    return 0;
}

/*
 * Guard macro for ensuring a valid data "kind" is being used for the
 * real domain routines.
 * Place at the top of all routines with switches on "kind" that accept
 * one of [s,d].
 */
#define ENSURE_VALID_REAL_KIND(__KIND) \
if (check_real_kind( __KIND ))         \
{                                 \
    return -1;                    \
}                                 \


/*
 * Checks that the char kind is valid for the complex domain (one of [c,z])
 * for use in blas/lapack.
 * Returns zero on success for status checking.
 */
static int check_complex_kind(char kind)
{
    switch (kind)
    {
        case 'c':
        case 'z':
            break;
        default:
        {
            PyGILState_STATE st = PyGILState_Ensure();
            PyErr_SetString(PyExc_ValueError,
                            "invalid data type (kind) found");
            PyGILState_Release(st);
        }
        return 1;
    }
    return 0;
}

/*
 * Guard macro for ensuring a valid data "kind" is being used for the
 * real domain routines.
 * Place at the top of all routines with switches on "kind" that accept
 * one of [c,z].
 */
#define ENSURE_VALID_COMPLEX_KIND(__KIND) \
if (check_complex_kind( __KIND ))         \
{                                 \
    return -1;                    \
}                                 \


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
NUMBA_EXPORT_FUNC(int)
numba_xxdot(char kind, char conjugate, Py_ssize_t n, void *dx, void *dy,
            void *result)
{
    void *raw_func = NULL;
    F_INT _n;
    F_INT inc = 1;

    ENSURE_VALID_KIND(kind)

    switch (kind)
    {
        case 's':
            raw_func = get_cblas_sdot();
            break;
        case 'd':
            raw_func = get_cblas_ddot();
            break;
        case 'c':
            raw_func = conjugate ? get_cblas_cdotc() : get_cblas_cdotu();
            break;
        case 'z':
            raw_func = conjugate ? get_cblas_zdotc() : get_cblas_zdotu();
            break;
    }
    if (raw_func == NULL)
        return -1;

    _n = (F_INT) n;

    switch (kind)
    {
        case 's':
            *(float *) result = (*(sdot_t) raw_func)(&_n, dx, &inc, dy, &inc);;
            break;
        case 'd':
            *(double *) result = (*(ddot_t) raw_func)(&_n, dx, &inc, dy, &inc);;
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
NUMBA_EXPORT_FUNC(int)
numba_xxgemv(char kind, char *trans, Py_ssize_t m, Py_ssize_t n,
             void *alpha, void *a, Py_ssize_t lda,
             void *x, void *beta, void *y)
{
    void *raw_func = NULL;
    F_INT _m, _n;
    F_INT _lda;
    F_INT inc = 1;

    ENSURE_VALID_KIND(kind)

    switch (kind)
    {
        case 's':
            raw_func = get_cblas_sgemv();
            break;
        case 'd':
            raw_func = get_cblas_dgemv();
            break;
        case 'c':
            raw_func = get_cblas_cgemv();
            break;
        case 'z':
            raw_func = get_cblas_zgemv();
            break;
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
NUMBA_EXPORT_FUNC(int)
numba_xxgemm(char kind, char *transa, char *transb,
             Py_ssize_t m, Py_ssize_t n, Py_ssize_t k,
             void *alpha, void *a, Py_ssize_t lda,
             void *b, Py_ssize_t ldb, void *beta,
             void *c, Py_ssize_t ldc)
{
    void *raw_func = NULL;
    F_INT _m, _n, _k;
    F_INT _lda, _ldb, _ldc;

    ENSURE_VALID_KIND(kind)

    switch (kind)
    {
        case 's':
            raw_func = get_cblas_sgemm();
            break;
        case 'd':
            raw_func = get_cblas_dgemm();
            break;
        case 'c':
            raw_func = get_cblas_cgemm();
            break;
        case 'z':
            raw_func = get_cblas_zgemm();
            break;
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

// Computes QR decompositions
EMIT_GET_CLAPACK_FUNC(sgeqrf)
EMIT_GET_CLAPACK_FUNC(dgeqrf)
EMIT_GET_CLAPACK_FUNC(cgeqrf)
EMIT_GET_CLAPACK_FUNC(zgeqrf)

// Computes columns of Q from elementary reflectors produced by xgeqrf() (QR).
EMIT_GET_CLAPACK_FUNC(sorgqr)
EMIT_GET_CLAPACK_FUNC(dorgqr)
EMIT_GET_CLAPACK_FUNC(cungqr)
EMIT_GET_CLAPACK_FUNC(zungqr)



#undef EMIT_GET_CLAPACK_FUNC

typedef void (*xxgetrf_t)(F_INT *m, F_INT *n, void *a, F_INT *lda, F_INT *ipiv,
                          F_INT *info);

typedef void (*xxgetri_t)(F_INT *n, void *a, F_INT *lda, F_INT *ipiv, void
                          *work, F_INT *lwork, F_INT *info);

typedef void (*xxpotrf_t)(char *uplo, F_INT *n, void *a, F_INT *lda, F_INT
                          *info);

typedef void (*rgeev_t)(char *jobvl, char *jobvr, F_INT *n, void *a, F_INT *lda,
                        void *wr, void *wi, void *vl, F_INT *ldvl, void *vr,
                        F_INT *ldvr, void *work, F_INT *lwork, F_INT *info);

typedef void (*cgeev_t)(char *jobvl, char *jobvr, F_INT *n, void *a, F_INT
                        *lda, void *w, void *vl, F_INT *ldvl, void *vr,
                        F_INT *ldvr, void *work, F_INT *lwork, void *rwork,
                        F_INT *info);

typedef void (*rgesdd_t)(char *jobz, F_INT *m, F_INT *n, void *a, F_INT *lda,
                         void *s, void *u, F_INT *ldu, void *vt, F_INT *ldvt,
                         void *work, F_INT *lwork, F_INT *iwork, F_INT *info);

typedef void (*cgesdd_t)(char *jobz, F_INT *m, F_INT *n, void *a, F_INT *lda,
                         void *s, void * u, F_INT *ldu, void * vt, F_INT *ldvt,
                         void *work, F_INT *lwork, void *rwork, F_INT *iwork,
                         F_INT *info);

typedef void (*xgeqrf_t)(F_INT *m, F_INT *n, void *a, F_INT *lda, void *tau,
                         void *work, F_INT *lwork, F_INT *info);

typedef void (*xxxgqr_t)(F_INT *m, F_INT *n, F_INT *k, void *a, F_INT *lda,
                         void *tau, void *work, F_INT *lwork, F_INT *info);



#define CATCH_LAPACK_INVALID_ARG(__routine, info)                      \
    do {                                                               \
        if (info < 0) {                                                \
            PyGILState_STATE st = PyGILState_Ensure();                 \
            PyErr_Format(PyExc_RuntimeError,                           \
                 "LAPACK Error: Routine " #__routine ". On input %d\n",\
                  -(int) info);                                        \
            PyGILState_Release(st);                                    \
            return -1;                                                 \
        }                                                              \
    } while(0)

/* Compute LU decomposition of A
 * NOTE: ipiv is an array of Fortran integers allocated by the caller,
 * which is therefore expected to use the right dtype.
 */
NUMBA_EXPORT_FUNC(int)
numba_xxgetrf(char kind, Py_ssize_t m, Py_ssize_t n, void *a, Py_ssize_t lda,
              F_INT *ipiv, Py_ssize_t *info)
{
    void *raw_func = NULL;
    F_INT _m, _n, _lda, _info;

    ENSURE_VALID_KIND(kind)

    switch (kind)
    {
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
    }
    if (raw_func == NULL)
        return -1;

    _m = (F_INT) m;
    _n = (F_INT) n;
    _lda = (F_INT) lda;

    (*(xxgetrf_t) raw_func)(&_m, &_n, a, &_lda, ipiv, &_info);
    *info = (Py_ssize_t) _info;
    return 0;
}

/* Compute the inverse of a matrix given its LU decomposition
 * (about ipiv, see numba_xxgetrf() above)
 */
NUMBA_EXPORT_FUNC(int)
numba_xxgetri(char kind, Py_ssize_t n, void *a, Py_ssize_t lda,
              F_INT *ipiv, void *work, Py_ssize_t *lwork,
              Py_ssize_t *info)
{
    void *raw_func = NULL;
    F_INT _n, _lda, _lwork, _info;

    ENSURE_VALID_KIND(kind)

    switch (kind)
    {
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
    }
    if (raw_func == NULL)
        return -1;

    _n = (F_INT) n;
    _lda = (F_INT) lda;
    _lwork = (F_INT) lwork[0]; // why is this a ptr?

    (*(xxgetri_t) raw_func)(&_n, a, &_lda, ipiv, work, &_lwork, &_info);
    *info = (Py_ssize_t) _info;
    return 0;
}

/* Compute the Cholesky factorization of a matrix.
 * Return -1 on internal error, 0 on success, > 0 on failure.
 */
NUMBA_EXPORT_FUNC(int)
numba_xxpotrf(char kind, char uplo, Py_ssize_t n, void *a, Py_ssize_t lda)
{
    void *raw_func = NULL;
    F_INT _n, _lda, info;

    ENSURE_VALID_KIND(kind)

    switch (kind)
    {
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
    }
    if (raw_func == NULL)
        return -1;

    _n = (F_INT) n;
    _lda = (F_INT) lda;

    (*(xxpotrf_t) raw_func)(&uplo, &_n, a, &_lda, &info);
    CATCH_LAPACK_INVALID_ARG("xxpotrf", info);
    return info;
}


/*
 * kind_size()
 * gets the data size appropriate for a specified kind.
 *
 * Input:
 * kind - the kind, one of:
 *         (s, d, c, z) = (float, double, complex, double complex).
 *
 * data_size - modified in place, on return contains the appropriate data size.
 *
 * Returns 0 on success, -1 else.
 */
static size_t kind_size(char kind)
{
    size_t data_size = 0;
    switch (kind)
    {
        case 's':
            data_size  = sizeof(float);
            break;
        case 'd':
            data_size  = sizeof(double);
            break;
        case 'c':
            data_size  = sizeof(npy_complex64);
            break;
        case 'z':
            data_size  = sizeof(npy_complex128);
            break;
    }
    return data_size;

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
static F_INT
cast_from_X(char kind, void *val)
{
    switch(kind)
    {
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
            PyErr_SetString(PyExc_ValueError,
                            "invalid kind in cast");
            PyGILState_Release(st);
        }
    }
    return -1;
}

static int
ez_geev_return(Py_ssize_t info)
{
    if (info > 0)
    {
        PyGILState_STATE st = PyGILState_Ensure();
        PyErr_Format(PyExc_ValueError,
                     "LAPACK Error: Failed to compute all "
                     "eigenvalues, no eigenvectors have been computed.\n "
                     "i+1:n of wr/wi contains converged eigenvalues. "
                     "i = %d (Fortran indexing)\n", (int) info);
        PyGILState_Release(st);
        return -1;
    }
    return (int) info;
}

// real space eigen systems info from dgeev/sgeev
static int
numba_raw_rgeev(char kind, char jobvl, char jobvr,
                Py_ssize_t n, void *a, Py_ssize_t lda, void *wr, void *wi,
                void *vl, Py_ssize_t ldvl, void *vr, Py_ssize_t ldvr,
                void *work, Py_ssize_t lwork, Py_ssize_t *info)
{
    void *raw_func = NULL;
    F_INT _n, _lda, _ldvl, _ldvr, _lwork, _info;

    ENSURE_VALID_REAL_KIND(kind)

    switch (kind)
    {
        case 's':
            raw_func = get_clapack_sgeev();
            break;
        case 'd':
            raw_func = get_clapack_dgeev();
            break;
    }
    if (raw_func == NULL)
        return -1;

    _n = (F_INT) n;
    _lda = (F_INT) lda;
    _ldvl = (F_INT) ldvl;
    _ldvr = (F_INT) ldvr;
    _lwork = (F_INT) lwork;

    (*(rgeev_t) raw_func)(&jobvl, &jobvr, &_n, a, &_lda, wr, wi, vl, &_ldvl, vr,
                          &_ldvr, work, &_lwork, &_info);
    *info = (Py_ssize_t) _info;
    return 0;
}

// real space eigen systems info from dgeev/sgeev
// as numba_raw_rgeev but the allocation and error handling is done for the user
// Args are as per LAPACK.
NUMBA_EXPORT_FUNC(int)
numba_ez_rgeev(char kind, char jobvl, char jobvr, Py_ssize_t n, void *a,
               Py_ssize_t lda, void *wr, void *wi, void *vl, Py_ssize_t ldvl,
               void *vr, Py_ssize_t ldvr)
{
    Py_ssize_t info = 0;
    F_INT lwork = -1;
    F_INT _n, _lda, _ldvl, _ldvr;
    size_t base_size = -1;
    void * work = NULL;
    all_dtypes stack_slot;

    ENSURE_VALID_REAL_KIND(kind)

    _n = (F_INT) n;
    _lda = (F_INT) lda;
    _ldvl = (F_INT) ldvl;
    _ldvr = (F_INT) ldvr;

    // decide on a base type size
    base_size = kind_size(kind);

    work = &stack_slot;
    numba_raw_rgeev(kind, jobvl, jobvr, _n, a, _lda, wr, wi, vl, _ldvl,
                    vr, _ldvr, work, lwork, &info);
    CATCH_LAPACK_INVALID_ARG("numba_raw_rgeev", info);

    lwork = cast_from_X(kind, work);
    if (checked_PyMem_RawMalloc(&work, base_size * lwork))
    {
        return -1;
    }
    numba_raw_rgeev(kind, jobvl, jobvr, _n, a, _lda, wr, wi, vl, _ldvl,
                    vr, _ldvr, work, lwork, &info);
    PyMem_RawFree(work);

    CATCH_LAPACK_INVALID_ARG("numba_raw_rgeev", info);

    return ez_geev_return(info);
}

// complex space eigen systems info from cgeev/zgeev
// Args are as per LAPACK.
static int
numba_raw_cgeev(char kind, char jobvl, char jobvr,
                Py_ssize_t n, void *a, Py_ssize_t lda, void *w, void *vl,
                Py_ssize_t ldvl, void *vr, Py_ssize_t ldvr, void *work,
                Py_ssize_t lwork, void *rwork, Py_ssize_t *info)
{
    void *raw_func = NULL;
    F_INT _n, _lda, _ldvl, _ldvr, _lwork, _info;

    ENSURE_VALID_COMPLEX_KIND(kind)

    _n = (F_INT) n;
    _lda = (F_INT) lda;
    _ldvl = (F_INT) ldvl;
    _ldvr = (F_INT) ldvr;
    _lwork = (F_INT) lwork;

    switch (kind)
    {
        case 'c':
            raw_func = get_clapack_cgeev();
            break;
        case 'z':
            raw_func = get_clapack_zgeev();
            break;
    }
    if (raw_func == NULL)
        return -1;

    (*(cgeev_t) raw_func)(&jobvl, &jobvr, &_n, a, &_lda, w, vl, &_ldvl, vr,
                          &_ldvr, work, &_lwork, rwork, &_info);
    *info = (Py_ssize_t) _info;
    return 0;
}


// complex space eigen systems info from cgeev/zgeev
// as numba_raw_cgeev but the allocation and error handling is done for the user
// Args are as per LAPACK.
NUMBA_EXPORT_FUNC(int)
numba_ez_cgeev(char kind, char jobvl, char jobvr,  Py_ssize_t n, void *a,
               Py_ssize_t lda, void *w, void *vl, Py_ssize_t ldvl, void *vr,
               Py_ssize_t ldvr)
{
    Py_ssize_t info = 0;
    F_INT lwork = -1;
    F_INT _n, _lda, _ldvl, _ldvr;
    size_t base_size = -1;
    all_dtypes stack_slot, wk;
    void * work = NULL;
    void * rwork = (void *)&wk;

    ENSURE_VALID_COMPLEX_KIND(kind)

    _n = (F_INT) n;
    _lda = (F_INT) lda;
    _ldvl = (F_INT) ldvl;
    _ldvr = (F_INT) ldvr;

    // decide on a base type size
    base_size = kind_size(kind);

    work = &stack_slot;
    numba_raw_cgeev(kind, jobvl, jobvr, n, a, lda, w, vl, ldvl,
                    vr, ldvr, work, lwork, rwork, &info);
    CATCH_LAPACK_INVALID_ARG("numba_raw_cgeev", info);

    lwork = cast_from_X(kind, work);
    if (checked_PyMem_RawMalloc((void**)&rwork, 2*n*base_size))
    {
        return -1;
    }
    if (checked_PyMem_RawMalloc(&work, base_size * lwork))
    {
        PyMem_RawFree(rwork);
        return -1;
    }
    numba_raw_cgeev(kind, jobvl, jobvr, _n, a, _lda, w, vl, _ldvl,
                    vr, _ldvr, work, lwork, rwork, &info);
    PyMem_RawFree(work);
    PyMem_RawFree(rwork);
    CATCH_LAPACK_INVALID_ARG("numba_raw_cgeev", info);

    return ez_geev_return(info);
}


static int
ez_gesdd_return(Py_ssize_t info)
{
    if (info > 0)
    {
        PyGILState_STATE st = PyGILState_Ensure();
        PyErr_Format(PyExc_ValueError,
                     "LAPACK Error: Convergence of internal algorithm "
                     "reported failure. \nThere were %d superdiagonal "
                     "elements that failed to converge.", (int) info);
        PyGILState_Release(st);
        return -1;
    }
    return (int) info;
}

// real space svd systems info from dgesdd/sgesdd
// Args are as per LAPACK.
static int
numba_raw_rgesdd(char kind, char jobz, Py_ssize_t m, Py_ssize_t n, void *a,
                 Py_ssize_t lda, void *s, void *u, Py_ssize_t ldu, void *vt,
                 Py_ssize_t ldvt, void *work, Py_ssize_t lwork,
                 F_INT *iwork, Py_ssize_t *info)
{
    void *raw_func = NULL;
    F_INT _m, _n, _lda, _ldu, _ldvt, _lwork, _info;

    ENSURE_VALID_REAL_KIND(kind)

    _m = (F_INT) m;
    _n = (F_INT) n;
    _lda = (F_INT) lda;
    _ldu = (F_INT) ldu;
    _ldvt = (F_INT) ldvt;
    _lwork = (F_INT) lwork;

    switch (kind)
    {
        case 's':
            raw_func = get_clapack_sgesdd();
            break;
        case 'd':
            raw_func = get_clapack_dgesdd();
            break;
    }
    if (raw_func == NULL)
        return -1;

    (*(rgesdd_t) raw_func)(&jobz, &_m, &_n, a, &_lda, s, u, &_ldu, vt, &_ldvt,
                           work, &_lwork, iwork, &_info);
    *info = (Py_ssize_t) _info;
    return 0;
}

// real space svd info from dgesdd/sgesdd.
// As numba_raw_rgesdd but the allocation and error handling is done for the
// user
// Args are as per LAPACK.
static int
numba_ez_rgesdd(char kind, char jobz, Py_ssize_t m, Py_ssize_t n, void *a,
                Py_ssize_t lda, void *s, void *u, Py_ssize_t ldu, void *vt,
                Py_ssize_t ldvt)
{
    Py_ssize_t info = 0;
    Py_ssize_t minmn = -1;
    Py_ssize_t lwork = -1;
    all_dtypes stack_slot, wk;
    size_t base_size = -1;
    F_INT *iwork = (F_INT *)&wk;
    void *work = NULL;

    ENSURE_VALID_REAL_KIND(kind)

    // decide on a base type size
    base_size = kind_size(kind);

    work = &stack_slot;

    /* Compute optimal work size (lwork) */
    numba_raw_rgesdd(kind, jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work,
                     lwork, iwork, &info);
    CATCH_LAPACK_INVALID_ARG("numba_raw_rgesdd", info);

    /* Allocate work array */
    lwork = cast_from_X(kind, work);
    if (checked_PyMem_RawMalloc(&work, base_size * lwork))
        return -1;
    minmn = m > n ? n : m;
    if (checked_PyMem_RawMalloc((void**) &iwork, 8 * minmn * sizeof(F_INT)))
    {
        PyMem_RawFree(work);
        return -1;
    }
    numba_raw_rgesdd(kind, jobz, m, n, a, lda, s, u ,ldu, vt, ldvt, work, lwork,
                     iwork, &info);
    PyMem_RawFree(work);
    PyMem_RawFree(iwork);
    CATCH_LAPACK_INVALID_ARG("numba_raw_rgesdd", info);

    return ez_gesdd_return(info);
}

// complex space svd systems info from cgesdd/zgesdd
// Args are as per LAPACK.
static int
numba_raw_cgesdd(char kind, char jobz, Py_ssize_t m, Py_ssize_t n, void *a,
                 Py_ssize_t lda, void *s, void *u, Py_ssize_t ldu, void *vt,
                 Py_ssize_t ldvt, void *work, Py_ssize_t lwork, void *rwork,
                 F_INT *iwork, Py_ssize_t *info)
{
    void *raw_func = NULL;
    F_INT _m, _n, _lda, _ldu, _ldvt, _lwork, _info;

    ENSURE_VALID_COMPLEX_KIND(kind)

    _m = (F_INT) m;
    _n = (F_INT) n;
    _lda = (F_INT) lda;
    _ldu = (F_INT) ldu;
    _ldvt = (F_INT) ldvt;
    _lwork = (F_INT) lwork;

    switch (kind)
    {
        case 'c':
            raw_func = get_clapack_cgesdd();
            break;
        case 'z':
            raw_func = get_clapack_zgesdd();
            break;
    }
    if (raw_func == NULL)
        return -1;

    (*(cgesdd_t) raw_func)(&jobz, &_m, &_n, a, &_lda, s, u, &_ldu, vt, &_ldvt,
                           work, &_lwork, rwork, iwork, &_info);
    *info = (Py_ssize_t) _info;
    return 0;
}

// complex space svd info from cgesdd/zgesdd.
// As numba_raw_cgesdd but the allocation and error handling is done for the
// user
// Args are as per LAPACK.
static int
numba_ez_cgesdd(char kind, char jobz, Py_ssize_t m, Py_ssize_t n, void *a,
                Py_ssize_t lda, void *s, void *u, Py_ssize_t ldu, void *vt,
                Py_ssize_t ldvt)
{
    Py_ssize_t info = 0;
    Py_ssize_t lwork = -1;
    Py_ssize_t lrwork = -1;
    Py_ssize_t minmn = -1;
    Py_ssize_t tmp1, tmp2;
    Py_ssize_t maxmn = -1;
    size_t real_base_size = -1;
    size_t complex_base_size = -1;
    all_dtypes stack_slot, wk1, wk2;
    void *work = NULL;
    void *rwork = (void *)&wk1;
    F_INT *iwork = (F_INT *)&wk2;

    ENSURE_VALID_COMPLEX_KIND(kind)

    // find the function to call, decide on a base type size
    switch (kind)
    {
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

    work = &stack_slot;

    /* Compute optimal work size (lwork) */
    numba_raw_cgesdd(kind, jobz, m, n, a, lda, s, u ,ldu, vt, ldvt, work, lwork,
                     rwork, iwork, &info);
    CATCH_LAPACK_INVALID_ARG("numba_raw_cgesdd", info);

    /* Allocate work array */
    lwork = cast_from_X(kind, work);
    if (checked_PyMem_RawMalloc(&work, complex_base_size * lwork))
        return -1;

    minmn = m > n ? n : m;
    if (jobz == 'n')
    {
        lrwork = 7 * minmn;
    }
    else
    {
        maxmn = m > n ? m : n;
        tmp1 = 5 * minmn + 7;
        tmp2 = 2 * maxmn + 2 * minmn + 1;
        lrwork = minmn * (tmp1 > tmp2 ? tmp1: tmp2);
    }

    if (checked_PyMem_RawMalloc(&rwork,
                                real_base_size * (lrwork > 1 ? lrwork : 1)))
    {
        PyMem_RawFree(work);
        return -1;
    }
    if (checked_PyMem_RawMalloc((void **) &iwork,
                                8 * minmn * sizeof(F_INT)))
    {
        PyMem_RawFree(work);
        PyMem_RawFree(rwork);
        return -1;
    }
    numba_raw_cgesdd(kind, jobz, m, n, a, lda, s, u ,ldu, vt, ldvt, work, lwork,
                     rwork, iwork, &info);
    PyMem_RawFree(work);
    PyMem_RawFree(rwork);
    PyMem_RawFree(iwork);
    CATCH_LAPACK_INVALID_ARG("numba_raw_cgesdd", info);

    return ez_gesdd_return(info);
}


// SVD systems info from *gesdd.
// This routine hides the type and general complexity involved with making the
// calls to *gesdd. The work space computation and error handling etc is hidden.
// Args are as per LAPACK.
NUMBA_EXPORT_FUNC(int)
numba_ez_gesdd(char kind, char jobz, Py_ssize_t m, Py_ssize_t n, void *a,
               Py_ssize_t lda, void *s, void *u, Py_ssize_t ldu, void *vt,
               Py_ssize_t ldvt)
{
    ENSURE_VALID_KIND(kind)

    switch (kind)
    {
        case 's':
        case 'd':
            return numba_ez_rgesdd(kind, jobz, m, n, a, lda, s, u, ldu, vt,
                                   ldvt);
        case 'c':
        case 'z':
            return numba_ez_cgesdd(kind, jobz, m, n, a, lda, s, u, ldu, vt,
                                   ldvt);
    }
    return -1; // unreachable
}


/*
 * Compute the QR factorization of a matrix.
 * Return -1 on internal error, 0 on success, > 0 on failure.
 */
static int
numba_raw_xgeqrf(char kind, Py_ssize_t m, Py_ssize_t n, void *a, Py_ssize_t
                 lda, void *tau, void *work, Py_ssize_t lwork, Py_ssize_t *info)
{
    void *raw_func = NULL;
    F_INT _m, _n, _lda, _lwork, _info;

    ENSURE_VALID_KIND(kind)

    switch (kind)
    {
        case 's':
            raw_func = get_clapack_sgeqrf();
            break;
        case 'd':
            raw_func = get_clapack_dgeqrf();
            break;
        case 'c':
            raw_func = get_clapack_cgeqrf();
            break;
        case 'z':
            raw_func = get_clapack_zgeqrf();
            break;
    }
    if (raw_func == NULL)
        return -1;

    _m = (F_INT) m;
    _n = (F_INT) n;
    _lda = (F_INT) lda;
    _lwork = (F_INT) lwork;

    (*(xgeqrf_t) raw_func)(&_m, &_n, a, &_lda, tau, work, &_lwork, &_info);
    *info = (Py_ssize_t) _info;
    return 0;
}

/*
 * Compute the QR factorization of a matrix.
 * This routine hides the type and general complexity involved with making the
 * xgeqrf calls. The work space computation and error handling etc is hidden.
 * Args are as per LAPACK.
 */
NUMBA_EXPORT_FUNC(int)
numba_ez_geqrf(char kind, Py_ssize_t m, Py_ssize_t n, void *a, Py_ssize_t
               lda, void *tau)
{
    Py_ssize_t info = 0;
    Py_ssize_t lwork = -1;
    size_t base_size = -1;
    all_dtypes stack_slot;
    void *work = NULL;

    base_size = kind_size(kind);

    work = &stack_slot;

    /* Compute optimal work size (lwork) */
    numba_raw_xgeqrf(kind, m, n, a, lda, tau, work, lwork, &info);
    CATCH_LAPACK_INVALID_ARG("numba_raw_xgeqrf", info);

    /* Allocate work array */
    lwork = cast_from_X(kind, work);
    if (checked_PyMem_RawMalloc(&work, base_size * lwork))
        return -1;

    numba_raw_xgeqrf(kind, m, n, a, lda, tau, work, lwork, &info);
    PyMem_RawFree(work);
    CATCH_LAPACK_INVALID_ARG("numba_raw_xgeqrf", info);

    return 0; // info cannot be >0

}


/*
 * Compute the orthogonal Q matrix (in QR) from elementary relectors.
 * Return -1 on internal error, 0 on success, > 0 on failure.
 */
static int
numba_raw_xxxgqr(char kind, Py_ssize_t m, Py_ssize_t n, Py_ssize_t k, void *a,
                 Py_ssize_t lda, void *tau, void * work, Py_ssize_t lwork, Py_ssize_t *info)
{
    void *raw_func = NULL;
    F_INT _m, _n, _k, _lda, _lwork, _info;

    ENSURE_VALID_KIND(kind)

    switch (kind)
    {
        case 's':
            raw_func = get_clapack_sorgqr();
            break;
        case 'd':
            raw_func = get_clapack_dorgqr();
            break;
        case 'c':
            raw_func = get_clapack_cungqr();
            break;
        case 'z':
            raw_func = get_clapack_zungqr();
            break;
    }
    if (raw_func == NULL)
        return -1;

    _m = (F_INT) m;
    _n = (F_INT) n;
    _k = (F_INT) k;
    _lda = (F_INT) lda;
    _lwork = (F_INT) lwork;

    (*(xxxgqr_t) raw_func)(&_m, &_n, &_k, a, &_lda, tau, work, &_lwork, &_info);
    *info = (Py_ssize_t) _info;
    return 0;
}


/*
 * Compute the orthogonal Q matrix (in QR) from elementary reflectors.
 * This routine hides the type and general complexity involved with making the
 * x{or,un}qrf calls. The work space computation and error handling etc is
 * hidden. Args are as per LAPACK.
 */
NUMBA_EXPORT_FUNC(int)
numba_ez_xxgqr(char kind, Py_ssize_t m, Py_ssize_t n, Py_ssize_t k, void *a,
               Py_ssize_t lda, void *tau)
{
    Py_ssize_t info = 0;
    Py_ssize_t lwork = -1;
    size_t base_size = -1;
    all_dtypes stack_slot;
    void *work = NULL;


    work = &stack_slot;

    /* Compute optimal work size (lwork) */
    numba_raw_xxxgqr(kind, m, n, k, a, lda, tau, work, lwork, &info);
    CATCH_LAPACK_INVALID_ARG("numba_raw_xxxgqr", info);

    base_size = kind_size(kind);

    /* Allocate work array */
    lwork = cast_from_X(kind, work);
    if (checked_PyMem_RawMalloc(&work, base_size * lwork))
        return -1;

    numba_raw_xxxgqr(kind, m, n, k, a, lda, tau, work, lwork, &info);
    PyMem_RawFree(work);
    CATCH_LAPACK_INVALID_ARG("numba_raw_xxxgqr", info);

    return 0; // info cannot be >0

}
