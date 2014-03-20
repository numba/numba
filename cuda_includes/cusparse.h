/*
This file is for binding generation.
The file has been modified from cusparse_v2.h header.
The following macros are removed
    - CUSPARSEAPI

WARNING This file should never be distributed.
*/

/* CUSPARSE initialization and managment routines */
cusparseStatus_t  cusparseCreate(cusparseHandle_t *handle);
cusparseStatus_t  cusparseDestroy(cusparseHandle_t handle);
cusparseStatus_t  cusparseGetVersion(cusparseHandle_t handle, int *version);
cusparseStatus_t  cusparseSetStream(cusparseHandle_t handle, cudaStream_t streamId);

/* CUSPARSE type creation, destruction, set and get routines */
cusparseStatus_t  cusparseGetPointerMode(cusparseHandle_t handle, cusparsePointerMode_t *mode);
cusparseStatus_t  cusparseSetPointerMode(cusparseHandle_t handle, cusparsePointerMode_t mode);

/* sparse matrix descriptor */
/* When the matrix descriptor is created, its fields are initialized to:
   CUSPARSE_MATRIX_TYPE_GENERAL
   CUSPARSE_INDEX_BASE_ZERO
   All other fields are uninitialized
*/
cusparseStatus_t  cusparseCreateMatDescr(cusparseMatDescr_t *descrA);
cusparseStatus_t  cusparseDestroyMatDescr (cusparseMatDescr_t descrA);

cusparseStatus_t  cusparseSetMatType(cusparseMatDescr_t descrA, cusparseMatrixType_t type);
cusparseMatrixType_t  cusparseGetMatType(const cusparseMatDescr_t descrA);

cusparseStatus_t  cusparseSetMatFillMode(cusparseMatDescr_t descrA, cusparseFillMode_t fillMode);
cusparseFillMode_t  cusparseGetMatFillMode(const cusparseMatDescr_t descrA);

cusparseStatus_t  cusparseSetMatDiagType(cusparseMatDescr_t  descrA, cusparseDiagType_t diagType);
cusparseDiagType_t  cusparseGetMatDiagType(const cusparseMatDescr_t descrA);

cusparseStatus_t  cusparseSetMatIndexBase(cusparseMatDescr_t descrA, cusparseIndexBase_t base);
cusparseIndexBase_t  cusparseGetMatIndexBase(const cusparseMatDescr_t descrA);

/* sparse triangular solve */
cusparseStatus_t  cusparseCreateSolveAnalysisInfo(cusparseSolveAnalysisInfo_t *info);
cusparseStatus_t  cusparseDestroySolveAnalysisInfo(cusparseSolveAnalysisInfo_t info);
cusparseStatus_t  cusparseGetLevelInfo(cusparseHandle_t handle,
                                                  cusparseSolveAnalysisInfo_t info,
                                                  int *nlevels,
                                                  int **levelPtr,
                                                  int **levelInd);

/* hybrid (HYB) format */
cusparseStatus_t  cusparseCreateHybMat(cusparseHybMat_t *hybA);
cusparseStatus_t  cusparseDestroyHybMat(cusparseHybMat_t hybA);


/* --- Sparse Level 1 routines --- */

/* Description: Addition of a scalar multiple of a sparse vector x
   and a dense vector y. */
cusparseStatus_t  cusparseSaxpyi_v2(cusparseHandle_t handle,
                                               int nnz,
                                               const float *alpha,
                                               const float *xVal,
                                               const int *xInd,
                                               float *y,
                                               cusparseIndexBase_t idxBase);

cusparseStatus_t  cusparseDaxpyi_v2(cusparseHandle_t handle,
                                               int nnz,
                                               const double *alpha,
                                               const double *xVal,
                                               const int *xInd,
                                               double *y,
                                               cusparseIndexBase_t idxBase);

cusparseStatus_t  cusparseCaxpyi_v2(cusparseHandle_t handle,
                                               int nnz,
                                               const cuComplex *alpha,
                                               const cuComplex *xVal,
                                               const int *xInd,
                                               cuComplex *y,
                                               cusparseIndexBase_t idxBase);

cusparseStatus_t  cusparseZaxpyi_v2(cusparseHandle_t handle,
                                               int nnz,
                                               const cuDoubleComplex *alpha,
                                               const cuDoubleComplex *xVal,
                                               const int *xInd,
                                               cuDoubleComplex *y,
                                               cusparseIndexBase_t idxBase);

/* Description: dot product of a sparse vector x and a dense vector y. */
cusparseStatus_t  cusparseSdoti(cusparseHandle_t handle,
                                           int nnz,
                                           const float *xVal,
                                           const int *xInd,
                                           const float *y,
                                           float *resultDevHostPtr,
                                           cusparseIndexBase_t idxBase);

cusparseStatus_t  cusparseDdoti(cusparseHandle_t handle,
                                           int nnz,
                                           const double *xVal,
                                           const int *xInd,
                                           const double *y,
                                           double *resultDevHostPtr,
                                           cusparseIndexBase_t idxBase);

cusparseStatus_t  cusparseCdoti(cusparseHandle_t handle,
                                           int nnz,
                                           const cuComplex *xVal,
                                           const int *xInd,
                                           const cuComplex *y,
                                           cuComplex *resultDevHostPtr,
                                           cusparseIndexBase_t idxBase);

cusparseStatus_t  cusparseZdoti(cusparseHandle_t handle,
                                           int nnz,
                                           const cuDoubleComplex *xVal,
                                           const int *xInd,
                                           const cuDoubleComplex *y,
                                           cuDoubleComplex *resultDevHostPtr,
                                           cusparseIndexBase_t idxBase);

/* Description: dot product of complex conjugate of a sparse vector x
   and a dense vector y. */
cusparseStatus_t  cusparseCdotci(cusparseHandle_t handle,
                                            int nnz,
                                            const cuComplex *xVal,
                                            const int *xInd,
                                            const cuComplex *y,
                                            cuComplex *resultDevHostPtr,
                                            cusparseIndexBase_t idxBase);

cusparseStatus_t  cusparseZdotci(cusparseHandle_t handle,
                                            int nnz,
                                            const cuDoubleComplex *xVal,
                                            const int *xInd,
                                            const cuDoubleComplex *y,
                                            cuDoubleComplex *resultDevHostPtr,
                                            cusparseIndexBase_t idxBase);


/* Description: Gather of non-zero elements from dense vector y into
   sparse vector x. */
cusparseStatus_t  cusparseSgthr(cusparseHandle_t handle,
                                           int nnz,
                                           const float *y,
                                           float *xVal,
                                           const int *xInd,
                                           cusparseIndexBase_t idxBase);

cusparseStatus_t  cusparseDgthr(cusparseHandle_t handle,
                                           int nnz,
                                           const double *y,
                                           double *xVal,
                                           const int *xInd,
                                           cusparseIndexBase_t idxBase);

cusparseStatus_t  cusparseCgthr(cusparseHandle_t handle,
                                           int nnz,
                                           const cuComplex *y,
                                           cuComplex *xVal,
                                           const int *xInd,
                                           cusparseIndexBase_t idxBase);

cusparseStatus_t  cusparseZgthr(cusparseHandle_t handle,
                                           int nnz,
                                           const cuDoubleComplex *y,
                                           cuDoubleComplex *xVal,
                                           const int *xInd,
                                           cusparseIndexBase_t idxBase);

/* Description: Gather of non-zero elements from desne vector y into
   sparse vector x (also replacing these elements in y by zeros). */
cusparseStatus_t  cusparseSgthrz(cusparseHandle_t handle,
                                            int nnz,
                                            float *y,
                                            float *xVal,
                                            const int *xInd,
                                            cusparseIndexBase_t idxBase);

cusparseStatus_t  cusparseDgthrz(cusparseHandle_t handle,
                                            int nnz,
                                            double *y,
                                            double *xVal,
                                            const int *xInd,
                                            cusparseIndexBase_t idxBase);

cusparseStatus_t  cusparseCgthrz(cusparseHandle_t handle,
                                            int nnz,
                                            cuComplex *y,
                                            cuComplex *xVal,
                                            const int *xInd,
                                            cusparseIndexBase_t idxBase);

cusparseStatus_t  cusparseZgthrz(cusparseHandle_t handle,
                                            int nnz,
                                            cuDoubleComplex *y,
                                            cuDoubleComplex *xVal,
                                            const int *xInd,
                                            cusparseIndexBase_t idxBase);

/* Description: Scatter of elements of the sparse vector x into
   dense vector y. */
cusparseStatus_t  cusparseSsctr(cusparseHandle_t handle,
                                           int nnz,
                                           const float *xVal,
                                           const int *xInd,
                                           float *y,
                                           cusparseIndexBase_t idxBase);

cusparseStatus_t  cusparseDsctr(cusparseHandle_t handle,
                                           int nnz,
                                           const double *xVal,
                                           const int *xInd,
                                           double *y,
                                           cusparseIndexBase_t idxBase);

cusparseStatus_t  cusparseCsctr(cusparseHandle_t handle,
                                           int nnz,
                                           const cuComplex *xVal,
                                           const int *xInd,
                                           cuComplex *y,
                                           cusparseIndexBase_t idxBase);

cusparseStatus_t  cusparseZsctr(cusparseHandle_t handle,
                                           int nnz,
                                           const cuDoubleComplex *xVal,
                                           const int *xInd,
                                           cuDoubleComplex *y,
                                           cusparseIndexBase_t idxBase);

/* Description: Givens rotation, where c and s are cosine and sine,
   x and y are sparse and dense vectors, respectively. */
cusparseStatus_t  cusparseSroti_v2(cusparseHandle_t handle,
                                              int nnz,
                                              float *xVal,
                                              const int *xInd,
                                              float *y,
                                              const float *c,
                                              const float *s,
                                              cusparseIndexBase_t idxBase);

cusparseStatus_t  cusparseDroti_v2(cusparseHandle_t handle,
                                              int nnz,
                                              double *xVal,
                                              const int *xInd,
                                              double *y,
                                              const double *c,
                                              const double *s,
                                              cusparseIndexBase_t idxBase);


/* --- Sparse Level 2 routines --- */

/* Description: Matrix-vector multiplication  y = alpha * op(A) * x  + beta * y,
   where A is a sparse matrix in CSR storage format, x and y are dense vectors. */
cusparseStatus_t  cusparseScsrmv_v2(cusparseHandle_t handle,
                                               cusparseOperation_t transA,
                                               int m,
                                               int n,
                                               int nnz,
                                               const float *alpha,
                                               const cusparseMatDescr_t descrA,
                                               const float *csrValA,
                                               const int *csrRowPtrA,
                                               const int *csrColIndA,
                                               const float *x,
                                               const float *beta,
                                               float *y);

cusparseStatus_t  cusparseDcsrmv_v2(cusparseHandle_t handle,
                                               cusparseOperation_t transA,
                                               int m,
                                               int n,
                                               int nnz,
                                               const double *alpha,
                                               const cusparseMatDescr_t descrA,
                                               const double *csrValA,
                                               const int *csrRowPtrA,
                                               const int *csrColIndA,
                                               const double *x,
                                               const double *beta,
                                               double *y);

cusparseStatus_t  cusparseCcsrmv_v2(cusparseHandle_t handle,
                                               cusparseOperation_t transA,
                                               int m,
                                               int n,
                                               int nnz,
                                               const cuComplex *alpha,
                                               const cusparseMatDescr_t descrA,
                                               const cuComplex *csrValA,
                                               const int *csrRowPtrA,
                                               const int *csrColIndA,
                                               const cuComplex *x,
                                               const cuComplex *beta,
                                               cuComplex *y);

cusparseStatus_t  cusparseZcsrmv_v2(cusparseHandle_t handle,
                                               cusparseOperation_t transA,
                                               int m,
                                               int n,
                                               int nnz,
                                               const cuDoubleComplex *alpha,
                                               const cusparseMatDescr_t descrA,
                                               const cuDoubleComplex *csrValA,
                                               const int *csrRowPtrA,
                                               const int *csrColIndA,
                                               const cuDoubleComplex *x,
                                               const cuDoubleComplex *beta,
                                               cuDoubleComplex *y);

/* Description: Matrix-vector multiplication  y = alpha * op(A) * x  + beta * y,
   where A is a sparse matrix in HYB storage format, x and y are dense vectors. */
cusparseStatus_t  cusparseShybmv(cusparseHandle_t handle,
                                            cusparseOperation_t transA,
                                            const float *alpha,
                                            const cusparseMatDescr_t descrA,
                                            const cusparseHybMat_t hybA,
                                            const float *x,
                                            const float *beta,
                                            float *y);

cusparseStatus_t  cusparseDhybmv(cusparseHandle_t handle,
                                            cusparseOperation_t transA,
                                            const double *alpha,
                                            const cusparseMatDescr_t descrA,
                                            const cusparseHybMat_t hybA,
                                            const double *x,
                                            const double *beta,
                                            double *y);

cusparseStatus_t  cusparseChybmv(cusparseHandle_t handle,
                                            cusparseOperation_t transA,
                                            const cuComplex *alpha,
                                            const cusparseMatDescr_t descrA,
                                            const cusparseHybMat_t hybA,
                                            const cuComplex *x,
                                            const cuComplex *beta,
                                            cuComplex *y);

cusparseStatus_t  cusparseZhybmv(cusparseHandle_t handle,
                                            cusparseOperation_t transA,
                                            const cuDoubleComplex *alpha,
                                            const cusparseMatDescr_t descrA,
                                            const cusparseHybMat_t hybA,
                                            const cuDoubleComplex *x,
                                            const cuDoubleComplex *beta,
                                            cuDoubleComplex *y);

/* Description: Matrix-vector multiplication  y = alpha * op(A) * x  + beta * y,
   where A is a sparse matrix in BSR storage format, x and y are dense vectors. */
cusparseStatus_t  cusparseSbsrmv(cusparseHandle_t handle,
                                            cusparseDirection_t dirA,
                                            cusparseOperation_t transA,
                                            int mb,
                                            int nb,
                                            int nnzb,
                                            const float *alpha,
                                            const cusparseMatDescr_t descrA,
                                            const float *bsrValA,
                                            const int *bsrRowPtrA,
                                            const int *bsrColIndA,
                                            int  blockDim,
                                            const float *x,
                                            const float *beta,
                                            float *y);

cusparseStatus_t  cusparseDbsrmv(cusparseHandle_t handle,
                                            cusparseDirection_t dirA,
                                            cusparseOperation_t transA,
                                            int mb,
                                            int nb,
                                            int nnzb,
                                            const double *alpha,
                                            const cusparseMatDescr_t descrA,
                                            const double *bsrValA,
                                            const int *bsrRowPtrA,
                                            const int *bsrColIndA,
                                            int  blockDim,
                                            const double *x,
                                            const double *beta,
                                            double *y);

cusparseStatus_t  cusparseCbsrmv(cusparseHandle_t handle,
                                            cusparseDirection_t dirA,
                                            cusparseOperation_t transA,
                                            int mb,
                                            int nb,
                                            int nnzb,
                                            const cuComplex *alpha,
                                            const cusparseMatDescr_t descrA,
                                            const cuComplex *bsrValA,
                                            const int *bsrRowPtrA,
                                            const int *bsrColIndA,
                                            int  blockDim,
                                            const cuComplex *x,
                                            const cuComplex *beta,
                                            cuComplex *y);

cusparseStatus_t  cusparseZbsrmv(cusparseHandle_t handle,
                                            cusparseDirection_t dirA,
                                            cusparseOperation_t transA,
                                            int mb,
                                            int nb,
                                            int nnzb,
                                            const cuDoubleComplex *alpha,
                                            const cusparseMatDescr_t descrA,
                                            const cuDoubleComplex *bsrValA,
                                            const int *bsrRowPtrA,
                                            const int *bsrColIndA,
                                            int  blockDim,
                                            const cuDoubleComplex *x,
                                            const cuDoubleComplex *beta,
                                            cuDoubleComplex *y);

/* Description: Matrix-vector multiplication  y = alpha * op(A) * x  + beta * y,
   where A is a sparse matrix in extended BSR storage format, x and y are dense
   vectors. */
cusparseStatus_t  cusparseSbsrxmv(cusparseHandle_t handle,
                                             cusparseDirection_t dirA,
                                             cusparseOperation_t transA,
                                             int sizeOfMask,
                                             int mb,
                                             int nb,
                                             int nnzb,
                                             const float *alpha,
                                             const cusparseMatDescr_t descrA,
                                             const float *bsrValA,
                                             const int *bsrMaskPtrA,
                                             const int *bsrRowPtrA,
                                             const int *bsrEndPtrA,
                                             const int *bsrColIndA,
                                             int  blockDim,
                                             const float *x,
                                             const float *beta,
                                             float *y);


cusparseStatus_t  cusparseDbsrxmv(cusparseHandle_t handle,
                                             cusparseDirection_t dirA,
                                             cusparseOperation_t transA,
                                             int sizeOfMask,
                                             int mb,
                                             int nb,
                                             int nnzb,
                                             const double *alpha,
                                             const cusparseMatDescr_t descrA,
                                             const double *bsrValA,
                                             const int *bsrMaskPtrA,
                                             const int *bsrRowPtrA,
                                             const int *bsrEndPtrA,
                                             const int *bsrColIndA,
                                             int  blockDim,
                                             const double *x,
                                             const double *beta,
                                             double *y);

cusparseStatus_t  cusparseCbsrxmv(cusparseHandle_t handle,
                                             cusparseDirection_t dirA,
                                             cusparseOperation_t transA,
                                             int sizeOfMask,
                                             int mb,
                                             int nb,
                                             int nnzb,
                                             const cuComplex *alpha,
                                             const cusparseMatDescr_t descrA,
                                             const cuComplex *bsrValA,
                                             const int *bsrMaskPtrA,
                                             const int *bsrRowPtrA,
                                             const int *bsrEndPtrA,
                                             const int *bsrColIndA,
                                             int  blockDim,
                                             const cuComplex *x,
                                             const cuComplex *beta,
                                             cuComplex *y);


cusparseStatus_t  cusparseZbsrxmv(cusparseHandle_t handle,
                                             cusparseDirection_t dirA,
                                             cusparseOperation_t transA,
                                             int sizeOfMask,
                                             int mb,
                                             int nb,
                                             int nnzb,
                                             const cuDoubleComplex *alpha,
                                             const cusparseMatDescr_t descrA,
                                             const cuDoubleComplex *bsrValA,
                                             const int *bsrMaskPtrA,
                                             const int *bsrRowPtrA,
                                             const int *bsrEndPtrA,
                                             const int *bsrColIndA,
                                             int  blockDim,
                                             const cuDoubleComplex *x,
                                             const cuDoubleComplex *beta,
                                             cuDoubleComplex *y);

/* Description: Solution of triangular linear system op(A) * y = alpha * x,
   where A is a sparse matrix in CSR storage format, x and y are dense vectors. */
cusparseStatus_t  cusparseScsrsv_analysis_v2(cusparseHandle_t handle,
                                                        cusparseOperation_t transA,
                                                        int m,
                                                        int nnz,
                                                        const cusparseMatDescr_t descrA,
                                                        const float *csrValA,
                                                        const int *csrRowPtrA,
                                                        const int *csrColIndA,
                                                        cusparseSolveAnalysisInfo_t info);

cusparseStatus_t  cusparseDcsrsv_analysis_v2(cusparseHandle_t handle,
                                                        cusparseOperation_t transA,
                                                        int m,
                                                        int nnz,
                                                        const cusparseMatDescr_t descrA,
                                                        const double *csrValA,
                                                        const int *csrRowPtrA,
                                                        const int *csrColIndA,
                                                        cusparseSolveAnalysisInfo_t info);

cusparseStatus_t  cusparseCcsrsv_analysis_v2(cusparseHandle_t handle,
                                                        cusparseOperation_t transA,
                                                        int m,
                                                        int nnz,
                                                        const cusparseMatDescr_t descrA,
                                                        const cuComplex *csrValA,
                                                        const int *csrRowPtrA,
                                                        const int *csrColIndA,
                                                        cusparseSolveAnalysisInfo_t info);

cusparseStatus_t  cusparseZcsrsv_analysis_v2(cusparseHandle_t handle,
                                                        cusparseOperation_t transA,
                                                        int m,
                                                        int nnz,
                                                        const cusparseMatDescr_t descrA,
                                                        const cuDoubleComplex *csrValA,
                                                        const int *csrRowPtrA,
                                                        const int *csrColIndA,
                                                        cusparseSolveAnalysisInfo_t info);


cusparseStatus_t  cusparseScsrsv_solve_v2(cusparseHandle_t handle,
                                                     cusparseOperation_t transA,
                                                     int m,
                                                     const float *alpha,
                                                     const cusparseMatDescr_t descrA,
                                                     const float *csrValA,
                                                     const int *csrRowPtrA,
                                                     const int *csrColIndA,
                                                     cusparseSolveAnalysisInfo_t info,
                                                     const float *x,
                                                     float *y);

cusparseStatus_t  cusparseDcsrsv_solve_v2(cusparseHandle_t handle,
                                                     cusparseOperation_t transA,
                                                     int m,
                                                     const double *alpha,
                                                     const cusparseMatDescr_t descrA,
                                                     const double *csrValA,
                                                     const int *csrRowPtrA,
                                                     const int *csrColIndA,
                                                     cusparseSolveAnalysisInfo_t info,
                                                     const double *x,
                                                     double *y);

cusparseStatus_t  cusparseCcsrsv_solve_v2(cusparseHandle_t handle,
                                                     cusparseOperation_t transA,
                                                     int m,
                                                     const cuComplex *alpha,
                                                     const cusparseMatDescr_t descrA,
                                                     const cuComplex *csrValA,
                                                     const int *csrRowPtrA,
                                                     const int *csrColIndA,
                                                     cusparseSolveAnalysisInfo_t info,
                                                     const cuComplex *x,
                                                     cuComplex *y);

cusparseStatus_t  cusparseZcsrsv_solve_v2(cusparseHandle_t handle,
                                                     cusparseOperation_t transA,
                                                     int m,
                                                     const cuDoubleComplex *alpha,
                                                     const cusparseMatDescr_t descrA,
                                                     const cuDoubleComplex *csrValA,
                                                     const int *csrRowPtrA,
                                                     const int *csrColIndA,
                                                     cusparseSolveAnalysisInfo_t info,
                                                     const cuDoubleComplex *x,
                                                     cuDoubleComplex *y);

/* Description: Solution of triangular linear system op(A) * y = alpha * x,
   where A is a sparse matrix in HYB storage format, x and y are dense vectors. */
cusparseStatus_t  cusparseShybsv_analysis(cusparseHandle_t handle,
                                                     cusparseOperation_t transA,
                                                     const cusparseMatDescr_t descrA,
                                                     cusparseHybMat_t hybA,
                                                     cusparseSolveAnalysisInfo_t info);

cusparseStatus_t  cusparseDhybsv_analysis(cusparseHandle_t handle,
                                                     cusparseOperation_t transA,
                                                     const cusparseMatDescr_t descrA,
                                                     cusparseHybMat_t hybA,
                                                     cusparseSolveAnalysisInfo_t info);

cusparseStatus_t  cusparseChybsv_analysis(cusparseHandle_t handle,
                                                     cusparseOperation_t transA,
                                                     const cusparseMatDescr_t descrA,
                                                     cusparseHybMat_t hybA,
                                                     cusparseSolveAnalysisInfo_t info);

cusparseStatus_t  cusparseZhybsv_analysis(cusparseHandle_t handle,
                                                     cusparseOperation_t transA,
                                                     const cusparseMatDescr_t descrA,
                                                     cusparseHybMat_t hybA,
                                                     cusparseSolveAnalysisInfo_t info);

cusparseStatus_t  cusparseShybsv_solve(cusparseHandle_t handle,
                                                  cusparseOperation_t trans,
                                                  const float *alpha,
                                                  const cusparseMatDescr_t descra,
                                                  const cusparseHybMat_t hybA,
                                                  cusparseSolveAnalysisInfo_t info,
                                                  const float *x,
                                                  float *y);

cusparseStatus_t  cusparseChybsv_solve(cusparseHandle_t handle,
                                                  cusparseOperation_t trans,
                                                  const cuComplex *alpha,
                                                  const cusparseMatDescr_t descra,
                                                  const cusparseHybMat_t hybA,
                                                  cusparseSolveAnalysisInfo_t info,
                                                  const cuComplex *x,
                                                  cuComplex *y);

cusparseStatus_t  cusparseDhybsv_solve(cusparseHandle_t handle,
                                                  cusparseOperation_t trans,
                                                  const double *alpha,
                                                  const cusparseMatDescr_t descra,
                                                  const cusparseHybMat_t hybA,
                                                  cusparseSolveAnalysisInfo_t info,
                                                  const double *x,
                                                  double *y);

cusparseStatus_t  cusparseZhybsv_solve(cusparseHandle_t handle,
                                                  cusparseOperation_t trans,
                                                  const cuDoubleComplex *alpha,
                                                  const cusparseMatDescr_t descra,
                                                  const cusparseHybMat_t hybA,
                                                  cusparseSolveAnalysisInfo_t info,
                                                  const cuDoubleComplex *x,
                                                  cuDoubleComplex *y);


/* --- Sparse Level 3 routines --- */

/* Description: Matrix-matrix multiplication C = alpha * op(A) * B  + beta * C,
   where A is a sparse matrix, B and C are dense and usually tall matrices. */
cusparseStatus_t  cusparseScsrmm_v2(cusparseHandle_t handle,
                                               cusparseOperation_t transA,
                                               int m,
                                               int n,
                                               int k,
                                               int nnz,
                                               const float *alpha,
                                               const cusparseMatDescr_t descrA,
                                               const float  *csrValA,
                                               const int *csrRowPtrA,
                                               const int *csrColIndA,
                                               const float *B,
                                               int ldb,
                                               const float *beta,
                                               float *C,
                                               int ldc);

cusparseStatus_t  cusparseDcsrmm_v2(cusparseHandle_t handle,
                                               cusparseOperation_t transA,
                                               int m,
                                               int n,
                                               int k,
                                               int nnz,
                                               const double *alpha,
                                               const cusparseMatDescr_t descrA,
                                               const double *csrValA,
                                               const int *csrRowPtrA,
                                               const int *csrColIndA,
                                               const double *B,
                                               int ldb,
                                               const double *beta,
                                               double *C,
                                               int ldc);

cusparseStatus_t  cusparseCcsrmm_v2(cusparseHandle_t handle,
                                               cusparseOperation_t transA,
                                               int m,
                                               int n,
                                               int k,
                                               int nnz,
                                               const cuComplex *alpha,
                                               const cusparseMatDescr_t descrA,
                                               const cuComplex  *csrValA,
                                               const int *csrRowPtrA,
                                               const int *csrColIndA,
                                               const cuComplex *B,
                                               int ldb,
                                               const cuComplex *beta,
                                               cuComplex *C,
                                               int ldc);

cusparseStatus_t  cusparseZcsrmm_v2(cusparseHandle_t handle,
                                               cusparseOperation_t transA,
                                               int m,
                                               int n,
                                               int k,
                                               int nnz,
                                               const cuDoubleComplex *alpha,
                                               const cusparseMatDescr_t descrA,
                                               const cuDoubleComplex  *csrValA,
                                               const int *csrRowPtrA,
                                               const int *csrColIndA,
                                               const cuDoubleComplex *B,
                                               int ldb,
                                               const cuDoubleComplex *beta,
                                               cuDoubleComplex *C,
                                               int ldc);


cusparseStatus_t  cusparseScsrmm2(cusparseHandle_t handle,
                                            cusparseOperation_t transa,
                                            cusparseOperation_t transb,
                                            int m,
                                            int n,
                                            int k,
                                            int nnz,
                                            const float *alpha,
                                            const cusparseMatDescr_t descrA,
                                            const float *csrValA,
                                            const int *csrRowPtrA,
                                            const int *csrColIndA,
                                            const float *B,
                                            int ldb,
                                            const float *beta,
                                            float *C,
                                            int ldc);

cusparseStatus_t  cusparseDcsrmm2(cusparseHandle_t handle,
                                            cusparseOperation_t transa,
                                            cusparseOperation_t transb,
                                            int m,
                                            int n,
                                            int k,
                                            int nnz,
                                            const double *alpha,
                                            const cusparseMatDescr_t descrA,
                                            const double *csrValA,
                                            const int *csrRowPtrA,
                                            const int *csrColIndA,
                                            const double *B,
                                            int ldb,
                                            const double *beta,
                                            double *C,
                                            int ldc);

cusparseStatus_t  cusparseCcsrmm2(cusparseHandle_t handle,
                                            cusparseOperation_t transa,
                                            cusparseOperation_t transb,
                                            int m,
                                            int n,
                                            int k,
                                            int nnz,
                                            const cuComplex *alpha,
                                            const cusparseMatDescr_t descrA,
                                            const cuComplex *csrValA,
                                            const int *csrRowPtrA,
                                            const int *csrColIndA,
                                            const cuComplex *B,
                                            int ldb,
                                            const cuComplex *beta,
                                            cuComplex *C,
                                            int ldc);

cusparseStatus_t  cusparseZcsrmm2(cusparseHandle_t handle,
                                            cusparseOperation_t transa,
                                            cusparseOperation_t transb,
                                            int m,
                                            int n,
                                            int k,
                                            int nnz,
                                            const cuDoubleComplex *alpha,
                                            const cusparseMatDescr_t descrA,
                                            const cuDoubleComplex *csrValA,
                                            const int *csrRowPtrA,
                                            const int *csrColIndA,
                                            const cuDoubleComplex *B,
                                            int ldb,
                                            const cuDoubleComplex *beta,
                                            cuDoubleComplex *C,
                                            int ldc);


/* Description: Solution of triangular linear system op(A) * Y = alpha * X,
   with multiple right-hand-sides, where A is a sparse matrix in CSR storage
   format, X and Y are dense and usually tall matrices. */
cusparseStatus_t  cusparseScsrsm_analysis(cusparseHandle_t handle,
                                                     cusparseOperation_t transA,
                                                     int m,
                                                     int nnz,
                                                     const cusparseMatDescr_t descrA,
                                                     const float *csrValA,
                                                     const int *csrRowPtrA,
                                                     const int *csrColIndA,
                                                     cusparseSolveAnalysisInfo_t info);

cusparseStatus_t  cusparseDcsrsm_analysis(cusparseHandle_t handle,
                                                     cusparseOperation_t transA,
                                                     int m,
                                                     int nnz,
                                                     const cusparseMatDescr_t descrA,
                                                     const double *csrValA,
                                                     const int *csrRowPtrA,
                                                     const int *csrColIndA,
                                                     cusparseSolveAnalysisInfo_t info);

cusparseStatus_t  cusparseCcsrsm_analysis(cusparseHandle_t handle,
                                                     cusparseOperation_t transA,
                                                     int m,
                                                     int nnz,
                                                     const cusparseMatDescr_t descrA,
                                                     const cuComplex *csrValA,
                                                     const int *csrRowPtrA,
                                                     const int *csrColIndA,
                                                     cusparseSolveAnalysisInfo_t info);

cusparseStatus_t  cusparseZcsrsm_analysis(cusparseHandle_t handle,
                                                     cusparseOperation_t transA,
                                                     int m,
                                                     int nnz,
                                                     const cusparseMatDescr_t descrA,
                                                     const cuDoubleComplex *csrValA,
                                                     const int *csrRowPtrA,
                                                     const int *csrColIndA,
                                                     cusparseSolveAnalysisInfo_t info);


cusparseStatus_t  cusparseScsrsm_solve(cusparseHandle_t handle,
                                                  cusparseOperation_t transA,
                                                  int m,
                                                  int n,
                                                  const float *alpha,
                                                  const cusparseMatDescr_t descrA,
                                                  const float *csrValA,
                                                  const int *csrRowPtrA,
                                                  const int *csrColIndA,
                                                  cusparseSolveAnalysisInfo_t info,
                                                  const float *x,
                                                  int ldx,
                                                  float *y,
                                                  int ldy);

cusparseStatus_t  cusparseDcsrsm_solve(cusparseHandle_t handle,
                                                  cusparseOperation_t transA,
                                                  int m,
                                                  int n,
                                                  const double *alpha,
                                                  const cusparseMatDescr_t descrA,
                                                  const double *csrValA,
                                                  const int *csrRowPtrA,
                                                  const int *csrColIndA,
                                                  cusparseSolveAnalysisInfo_t info,
                                                  const double *x,
                                                  int ldx,
                                                  double *y,
                                                  int ldy);

cusparseStatus_t  cusparseCcsrsm_solve(cusparseHandle_t handle,
                                                  cusparseOperation_t transA,
                                                  int m,
                                                  int n,
                                                  const cuComplex *alpha,
                                                  const cusparseMatDescr_t descrA,
                                                  const cuComplex *csrValA,
                                                  const int *csrRowPtrA,
                                                  const int *csrColIndA,
                                                  cusparseSolveAnalysisInfo_t info,
                                                  const cuComplex *x,
                                                  int ldx,
                                                  cuComplex *y,
                                                  int ldy);

cusparseStatus_t  cusparseZcsrsm_solve(cusparseHandle_t handle,
                                                  cusparseOperation_t transA,
                                                  int m,
                                                  int n,
                                                  const cuDoubleComplex *alpha,
                                                  const cusparseMatDescr_t descrA,
                                                  const cuDoubleComplex *csrValA,
                                                  const int *csrRowPtrA,
                                                  const int *csrColIndA,
                                                  cusparseSolveAnalysisInfo_t info,
                                                  const cuDoubleComplex *x,
                                                  int ldx,
                                                  cuDoubleComplex *y,
                                                  int ldy);

/* --- Preconditioners --- */

/* Description: Compute the incomplete-LU factorization with 0 fill-in (ILU0)
   based on the information in the opaque structure info that was obtained
   from the analysis phase (csrsv_analysis). */
cusparseStatus_t  cusparseScsrilu0(cusparseHandle_t handle,
                                              cusparseOperation_t trans,
                                              int m,
                                              const cusparseMatDescr_t descrA,
                                              float *csrValA_ValM,
                                              /* matrix A values are updated inplace
                                                 to be the preconditioner M values */
                                              const int *csrRowPtrA,
                                              const int *csrColIndA,
                                              cusparseSolveAnalysisInfo_t info);

cusparseStatus_t  cusparseDcsrilu0(cusparseHandle_t handle,
                                              cusparseOperation_t trans,
                                              int m,
                                              const cusparseMatDescr_t descrA,
                                              double *csrValA_ValM,
                                              /* matrix A values are updated inplace
                                                 to be the preconditioner M values */
                                              const int *csrRowPtrA,
                                              const int *csrColIndA,
                                              cusparseSolveAnalysisInfo_t info);

cusparseStatus_t  cusparseCcsrilu0(cusparseHandle_t handle,
                                              cusparseOperation_t trans,
                                              int m,
                                              const cusparseMatDescr_t descrA,
                                              cuComplex *csrValA_ValM,
                                              /* matrix A values are updated inplace
                                                 to be the preconditioner M values */
                                              const int *csrRowPtrA,
                                              const int *csrColIndA,
                                              cusparseSolveAnalysisInfo_t info);

cusparseStatus_t  cusparseZcsrilu0(cusparseHandle_t handle,
                                              cusparseOperation_t trans,
                                              int m,
                                              const cusparseMatDescr_t descrA,
                                              cuDoubleComplex *csrValA_ValM,
                                              /* matrix A values are updated inplace
                                                 to be the preconditioner M values */
                                              const int *csrRowPtrA,
                                              const int *csrColIndA,
                                              cusparseSolveAnalysisInfo_t info);

/* Description: Compute the incomplete-Cholesky factorization with 0 fill-in (IC0)
   based on the information in the opaque structure info that was obtained
   from the analysis phase (csrsv_analysis). */
cusparseStatus_t  cusparseScsric0(cusparseHandle_t handle,
                                              cusparseOperation_t trans,
                                              int m,
                                              const cusparseMatDescr_t descrA,
                                              float *csrValA_ValM,
                                              /* matrix A values are updated inplace
                                                 to be the preconditioner M values */
                                              const int *csrRowPtrA,
                                              const int *csrColIndA,
                                              cusparseSolveAnalysisInfo_t info);

cusparseStatus_t  cusparseDcsric0(cusparseHandle_t handle,
                                              cusparseOperation_t trans,
                                              int m,
                                              const cusparseMatDescr_t descrA,
                                              double *csrValA_ValM,
                                              /* matrix A values are updated inplace
                                                 to be the preconditioner M values */
                                              const int *csrRowPtrA,
                                              const int *csrColIndA,
                                              cusparseSolveAnalysisInfo_t info);

cusparseStatus_t  cusparseCcsric0(cusparseHandle_t handle,
                                              cusparseOperation_t trans,
                                              int m,
                                              const cusparseMatDescr_t descrA,
                                              cuComplex *csrValA_ValM,
                                              /* matrix A values are updated inplace
                                                 to be the preconditioner M values */
                                              const int *csrRowPtrA,
                                              const int *csrColIndA,
                                              cusparseSolveAnalysisInfo_t info);

cusparseStatus_t  cusparseZcsric0(cusparseHandle_t handle,
                                              cusparseOperation_t trans,
                                              int m,
                                              const cusparseMatDescr_t descrA,
                                              cuDoubleComplex *csrValA_ValM,
                                              /* matrix A values are updated inplace
                                                 to be the preconditioner M values */
                                              const int *csrRowPtrA,
                                              const int *csrColIndA,
                                              cusparseSolveAnalysisInfo_t info);


/* Description: Solution of tridiagonal linear system A * B = B,
   with multiple right-hand-sides. The coefficient matrix A is
   composed of lower (dl), main (d) and upper (du) diagonals, and
   the right-hand-sides B are overwritten with the solution.
   These routine use pivoting */
cusparseStatus_t cusparseSgtsv(cusparseHandle_t handle,
                               int m,
                               int n,
                               const float *dl,
                               const float  *d,
                               const float *du,
                               float *B,
                               int ldb);

cusparseStatus_t cusparseDgtsv(cusparseHandle_t handle,
                               int m,
                               int n,
                               const double *dl,
                               const double  *d,
                               const double *du,
                               double *B,
                               int ldb);

cusparseStatus_t cusparseCgtsv(cusparseHandle_t handle,
                               int m,
                               int n,
                               const cuComplex *dl,
                               const cuComplex  *d,
                               const cuComplex *du,
                               cuComplex *B,
                               int ldb);

cusparseStatus_t cusparseZgtsv(cusparseHandle_t handle,
                               int m,
                               int n,
                               const cuDoubleComplex *dl,
                               const cuDoubleComplex  *d,
                               const cuDoubleComplex *du,
                               cuDoubleComplex *B,
                               int ldb);
/* Description: Solution of tridiagonal linear system A * B = B,
   with multiple right-hand-sides. The coefficient matrix A is
   composed of lower (dl), main (d) and upper (du) diagonals, and
   the right-hand-sides B are overwritten with the solution.
   These routines do not use pivoting, using a combination of PCR and CR algorithm */
cusparseStatus_t cusparseSgtsv_nopivot(cusparseHandle_t handle,
                               int m,
                               int n,
                               const float *dl,
                               const float  *d,
                               const float *du,
                               float *B,
                               int ldb);

cusparseStatus_t cusparseDgtsv_nopivot(cusparseHandle_t handle,
                               int m,
                               int n,
                               const double *dl,
                               const double  *d,
                               const double *du,
                               double *B,
                               int ldb);

cusparseStatus_t cusparseCgtsv_nopivot(cusparseHandle_t handle,
                               int m,
                               int n,
                               const cuComplex *dl,
                               const cuComplex  *d,
                               const cuComplex *du,
                               cuComplex *B,
                               int ldb);

cusparseStatus_t cusparseZgtsv_nopivot(cusparseHandle_t handle,
                               int m,
                               int n,
                               const cuDoubleComplex *dl,
                               const cuDoubleComplex  *d,
                               const cuDoubleComplex *du,
                               cuDoubleComplex *B,
                               int ldb);

/* Description: Solution of a set of tridiagonal linear systems
   A * x = x, each with a single right-hand-side. The coefficient
   matrices A are composed of lower (dl), main (d) and upper (du)
   diagonals and stored separated by a batchStride, while the
   right-hand-sides x are also separated by a batchStride. */
cusparseStatus_t cusparseSgtsvStridedBatch(cusparseHandle_t handle,
                                           int m,
                                           const float *dl,
                                           const float  *d,
                                           const float *du,
                                           float *x,
                                           int batchCount,
                                           int batchStride);


cusparseStatus_t cusparseDgtsvStridedBatch(cusparseHandle_t handle,
                                           int m,
                                           const double *dl,
                                           const double  *d,
                                           const double *du,
                                           double *x,
                                           int batchCount,
                                           int batchStride);

cusparseStatus_t cusparseCgtsvStridedBatch(cusparseHandle_t handle,
                                           int m,
                                           const cuComplex *dl,
                                           const cuComplex  *d,
                                           const cuComplex *du,
                                           cuComplex *x,
                                           int batchCount,
                                           int batchStride);

cusparseStatus_t cusparseZgtsvStridedBatch(cusparseHandle_t handle,
                                           int m,
                                           const cuDoubleComplex *dl,
                                           const cuDoubleComplex  *d,
                                           const cuDoubleComplex *du,
                                           cuDoubleComplex *x,
                                           int batchCount,
                                           int batchStride);

/* --- Extra --- */

/* Description: This routine computes a sparse matrix that results from
   multiplication of two sparse matrices. */
cusparseStatus_t  cusparseXcsrgemmNnz(cusparseHandle_t handle,
                                                 cusparseOperation_t transA,
                                                 cusparseOperation_t transB,
                                                 int m,
                                                 int n,
                                                 int k,
                                                 const cusparseMatDescr_t descrA,
                                                 const int nnzA,
                                                 const int *csrRowPtrA,
                                                 const int *csrColIndA,
                                                 const cusparseMatDescr_t descrB,
                                                 const int nnzB,
                                                 const int *csrRowPtrB,
                                                 const int *csrColIndB,
                                                 const cusparseMatDescr_t descrC,
                                                 int *csrRowPtrC,
                                                 int *nnzTotalDevHostPtr);

cusparseStatus_t  cusparseScsrgemm(cusparseHandle_t handle,
                                              cusparseOperation_t transA,
                                              cusparseOperation_t transB,
                                              int m,
                                              int n,
                                              int k,
                                              const cusparseMatDescr_t descrA,
                                              const int nnzA,
                                              const float *csrValA,
                                              const int *csrRowPtrA,
                                              const int *csrColIndA,
                                              const cusparseMatDescr_t descrB,
                                              const int nnzB,
                                              const float *csrValB,
                                              const int *csrRowPtrB,
                                              const int *csrColIndB,
                                              const cusparseMatDescr_t descrC,
                                              float *csrValC,
                                              const int *csrRowPtrC,
                                              int *csrColIndC);

cusparseStatus_t  cusparseDcsrgemm(cusparseHandle_t handle,
                                              cusparseOperation_t transA,
                                              cusparseOperation_t transB,
                                              int m,
                                              int n,
                                              int k,
                                              const cusparseMatDescr_t descrA,
                                              int nnzA,
                                              const double *csrValA,
                                              const int *csrRowPtrA,
                                              const int *csrColIndA,
                                              const cusparseMatDescr_t descrB,
                                              int nnzB,
                                              const double *csrValB,
                                              const int *csrRowPtrB,
                                              const int *csrColIndB,
                                              const cusparseMatDescr_t descrC,
                                              double *csrValC,
                                              const int *csrRowPtrC,
                                              int *csrColIndC);

cusparseStatus_t  cusparseCcsrgemm(cusparseHandle_t handle,
                                              cusparseOperation_t transA,
                                              cusparseOperation_t transB,
                                              int m,
                                              int n,
                                              int k,
                                              const cusparseMatDescr_t descrA,
                                              int nnzA,
                                              const cuComplex *csrValA,
                                              const int *csrRowPtrA,
                                              const int *csrColIndA,
                                              const cusparseMatDescr_t descrB,
                                              int nnzB,
                                              const cuComplex *csrValB,
                                              const int *csrRowPtrB,
                                              const int *csrColIndB,
                                              const cusparseMatDescr_t descrC,
                                              cuComplex *csrValC,
                                              const int *csrRowPtrC,
                                              int *csrColIndC);

cusparseStatus_t  cusparseZcsrgemm(cusparseHandle_t handle,
                                              cusparseOperation_t transA,
                                              cusparseOperation_t transB,
                                              int m,
                                              int n,
                                              int k,
                                              const cusparseMatDescr_t descrA,
                                              int nnzA,
                                              const cuDoubleComplex *csrValA,
                                              const int *csrRowPtrA,
                                              const int *csrColIndA,
                                              const cusparseMatDescr_t descrB,
                                              int nnzB,
                                              const cuDoubleComplex *csrValB,
                                              const int *csrRowPtrB,
                                              const int *csrColIndB,
                                              const cusparseMatDescr_t descrC,
                                              cuDoubleComplex *csrValC,
                                              const int *csrRowPtrC,
                                              int *csrColIndC);

/* Description: This routine computes a sparse matrix that results from
   addition of two sparse matrices. */
cusparseStatus_t  cusparseXcsrgeamNnz(cusparseHandle_t handle,
                                                 int m,
                                                 int n,
                                                 const cusparseMatDescr_t descrA,
                                                 int nnzA,
                                                 const int *csrRowPtrA,
                                                 const int *csrColIndA,
                                                 const cusparseMatDescr_t descrB,
                                                 int nnzB,
                                                 const int *csrRowPtrB,
                                                 const int *csrColIndB,
                                                 const cusparseMatDescr_t descrC,
                                                 int *csrRowPtrC,
                                                 int *nnzTotalDevHostPtr);

cusparseStatus_t  cusparseScsrgeam(cusparseHandle_t handle,
                                              int m,
                                              int n,
                                              const float *alpha,
                                              const cusparseMatDescr_t descrA,
                                              int nnzA,
                                              const float *csrValA,
                                              const int *csrRowPtrA,
                                              const int *csrColIndA,
                                              const float *beta,
                                              const cusparseMatDescr_t descrB,
                                              int nnzB,
                                              const float *csrValB,
                                              const int *csrRowPtrB,
                                              const int *csrColIndB,
                                              const cusparseMatDescr_t descrC,
                                              float *csrValC,
                                              int *csrRowPtrC,
                                              int *csrColIndC);

cusparseStatus_t  cusparseDcsrgeam(cusparseHandle_t handle,
                                              int m,
                                              int n,
                                              const double *alpha,
                                              const cusparseMatDescr_t descrA,
                                              int nnzA,
                                              const double *csrValA,
                                              const int *csrRowPtrA,
                                              const int *csrColIndA,
                                              const double *beta,
                                              const cusparseMatDescr_t descrB,
                                              int nnzB,
                                              const double *csrValB,
                                              const int *csrRowPtrB,
                                              const int *csrColIndB,
                                              const cusparseMatDescr_t descrC,
                                              double *csrValC,
                                              int *csrRowPtrC,
                                              int *csrColIndC);

cusparseStatus_t  cusparseCcsrgeam(cusparseHandle_t handle,
                                              int m,
                                              int n,
                                              const cuComplex *alpha,
                                              const cusparseMatDescr_t descrA,
                                              int nnzA,
                                              const cuComplex *csrValA,
                                              const int *csrRowPtrA,
                                              const int *csrColIndA,
                                              const cuComplex *beta,
                                              const cusparseMatDescr_t descrB,
                                              int nnzB,
                                              const cuComplex *csrValB,
                                              const int *csrRowPtrB,
                                              const int *csrColIndB,
                                              const cusparseMatDescr_t descrC,
                                              cuComplex *csrValC,
                                              int *csrRowPtrC,
                                              int *csrColIndC);

cusparseStatus_t  cusparseZcsrgeam(cusparseHandle_t handle,
                                              int m,
                                              int n,
                                              const cuDoubleComplex *alpha,
                                              const cusparseMatDescr_t descrA,
                                              int nnzA,
                                              const cuDoubleComplex *csrValA,
                                              const int *csrRowPtrA,
                                              const int *csrColIndA,
                                              const cuDoubleComplex *beta,
                                              const cusparseMatDescr_t descrB,
                                              int nnzB,
                                              const cuDoubleComplex *csrValB,
                                              const int *csrRowPtrB,
                                              const int *csrColIndB,
                                              const cusparseMatDescr_t descrC,
                                              cuDoubleComplex *csrValC,
                                              int *csrRowPtrC,
                                              int *csrColIndC);

/* --- Sparse Format Conversion --- */

/* Description: This routine finds the total number of non-zero elements and
   the number of non-zero elements per row or column in the dense matrix A. */
cusparseStatus_t  cusparseSnnz(cusparseHandle_t handle,
                                          cusparseDirection_t dirA,
                                          int m,
                                          int n,
                                          const cusparseMatDescr_t  descrA,
                                          const float *A,
                                          int lda,
                                          int *nnzPerRowCol,
                                          int *nnzTotalDevHostPtr);

cusparseStatus_t  cusparseDnnz(cusparseHandle_t handle,
                                          cusparseDirection_t dirA,
                                          int m,
                                          int n,
                                          const cusparseMatDescr_t  descrA,
                                          const double *A,
                                          int lda,
                                          int *nnzPerRowCol,
                                          int *nnzTotalDevHostPtr);

cusparseStatus_t  cusparseCnnz(cusparseHandle_t handle,
                                          cusparseDirection_t dirA,
                                          int m,
                                          int n,
                                          const cusparseMatDescr_t  descrA,
                                          const cuComplex *A,
                                          int lda,
                                          int *nnzPerRowCol,
                                          int *nnzTotalDevHostPtr);

cusparseStatus_t  cusparseZnnz(cusparseHandle_t handle,
                                          cusparseDirection_t dirA,
                                          int m,
                                          int n,
                                          const cusparseMatDescr_t  descrA,
                                          const cuDoubleComplex *A,
                                          int lda,
                                          int *nnzPerRowCol,
                                          int *nnzTotalDevHostPtr);

/* Description: This routine converts a dense matrix to a sparse matrix
   in the CSR storage format, using the information computed by the
   nnz routine. */
cusparseStatus_t  cusparseSdense2csr(cusparseHandle_t handle,
                                                int m,
                                                int n,
                                                const cusparseMatDescr_t descrA,
                                                const float *A,
                                                int lda,
                                                const int *nnzPerRow,
                                                float *csrValA,
                                                int *csrRowPtrA,
                                                int *csrColIndA);

cusparseStatus_t  cusparseDdense2csr(cusparseHandle_t handle,
                                                int m,
                                                int n,
                                                const cusparseMatDescr_t descrA,
                                                const double *A,
                                                int lda,
                                                const int *nnzPerRow,
                                                double *csrValA,
                                                int *csrRowPtrA,
                                                int *csrColIndA);

cusparseStatus_t  cusparseCdense2csr(cusparseHandle_t handle,
                                                int m,
                                                int n,
                                                const cusparseMatDescr_t descrA,
                                                const cuComplex *A,
                                                int lda,
                                                const int *nnzPerRow,
                                                cuComplex *csrValA,
                                                int *csrRowPtrA,
                                                int *csrColIndA);

cusparseStatus_t  cusparseZdense2csr(cusparseHandle_t handle,
                                                int m,
                                                int n,
                                                const cusparseMatDescr_t descrA,
                                                const cuDoubleComplex *A,
                                                int lda,
                                                const int *nnzPerRow,
                                                cuDoubleComplex *csrValA,
                                                int *csrRowPtrA,
                                                int *csrColIndA);

/* Description: This routine converts a sparse matrix in CSR storage format
   to a dense matrix. */
cusparseStatus_t  cusparseScsr2dense(cusparseHandle_t handle,
                                                int m,
                                                int n,
                                                const cusparseMatDescr_t descrA,
                                                const float *csrValA,
                                                const int *csrRowPtrA,
                                                const int *csrColIndA,
                                                float *A,
                                                int lda);

cusparseStatus_t  cusparseDcsr2dense(cusparseHandle_t handle,
                                                int m,
                                                int n,
                                                const cusparseMatDescr_t descrA,
                                                const double *csrValA,
                                                const int *csrRowPtrA,
                                                const int *csrColIndA,
                                                double *A,
                                                int lda);

cusparseStatus_t  cusparseCcsr2dense(cusparseHandle_t handle,
                                                int m,
                                                int n,
                                                const cusparseMatDescr_t descrA,
                                                const cuComplex *csrValA,
                                                const int *csrRowPtrA,
                                                const int *csrColIndA,
                                                cuComplex *A,
                                                int lda);

cusparseStatus_t  cusparseZcsr2dense(cusparseHandle_t handle,
                                                int m,
                                                int n,
                                                const cusparseMatDescr_t descrA,
                                                const cuDoubleComplex *csrValA,
                                                const int *csrRowPtrA,
                                                const int *csrColIndA,
                                                cuDoubleComplex *A,
                                                int lda);

/* Description: This routine converts a dense matrix to a sparse matrix
   in the CSC storage format, using the information computed by the
   nnz routine. */
cusparseStatus_t  cusparseSdense2csc(cusparseHandle_t handle,
                                                int m,
                                                int n,
                                                const cusparseMatDescr_t descrA,
                                                const float *A,
                                                int lda,
                                                const int *nnzPerCol,
                                                float *cscValA,
                                                int *cscRowIndA,
                                                int *cscColPtrA);

cusparseStatus_t  cusparseDdense2csc(cusparseHandle_t handle,
                                                int m,
                                                int n,
                                                const cusparseMatDescr_t descrA,
                                                const double *A,
                                                int lda,
                                                const int *nnzPerCol,
                                                double *cscValA,
                                                int *cscRowIndA,
                                                int *cscColPtrA);

cusparseStatus_t  cusparseCdense2csc(cusparseHandle_t handle,
                                                int m,
                                                int n,
                                                const cusparseMatDescr_t descrA,
                                                const cuComplex *A,
                                                int lda,
                                                const int *nnzPerCol,
                                                cuComplex *cscValA,
                                                int *cscRowIndA,
                                                int *cscColPtrA);

cusparseStatus_t  cusparseZdense2csc(cusparseHandle_t handle,
                                                int m,
                                                int n,
                                                const cusparseMatDescr_t descrA,
                                                const cuDoubleComplex *A,
                                                int lda,
                                                const int *nnzPerCol,
                                                cuDoubleComplex *cscValA,
                                                int *cscRowIndA,
                                                int *cscColPtrA);

/* Description: This routine converts a sparse matrix in CSC storage format
   to a dense matrix. */
cusparseStatus_t  cusparseScsc2dense(cusparseHandle_t handle,
                                                int m,
                                                int n,
                                                const cusparseMatDescr_t descrA,
                                                const float *cscValA,
                                                const int *cscRowIndA,
                                                const int *cscColPtrA,
                                                float *A,
                                                int lda);

cusparseStatus_t  cusparseDcsc2dense(cusparseHandle_t handle,
                                                int m,
                                                int n,
                                                const cusparseMatDescr_t descrA,
                                                const double *cscValA,
                                                const int *cscRowIndA,
                                                const int *cscColPtrA,
                                                double *A,
                                                int lda);

cusparseStatus_t  cusparseCcsc2dense(cusparseHandle_t handle,
                                                int m,
                                                int n,
                                                const cusparseMatDescr_t descrA,
                                                const cuComplex *cscValA,
                                                const int *cscRowIndA,
                                                const int *cscColPtrA,
                                                cuComplex *A,
                                                int lda);

cusparseStatus_t  cusparseZcsc2dense(cusparseHandle_t handle,
                                                int m,
                                                int n,
                                                const cusparseMatDescr_t descrA,
                                                const cuDoubleComplex *cscValA,
                                                const int *cscRowIndA,
                                                const int *cscColPtrA,
                                                cuDoubleComplex *A,
                                                int lda);

/* Description: This routine compresses the indecis of rows or columns.
   It can be interpreted as a conversion from COO to CSR sparse storage
   format. */
cusparseStatus_t  cusparseXcoo2csr(cusparseHandle_t handle,
                                              const int *cooRowInd,
                                              int nnz,
                                              int m,
                                              int *csrRowPtr,
                                              cusparseIndexBase_t idxBase);

/* Description: This routine uncompresses the indecis of rows or columns.
   It can be interpreted as a conversion from CSR to COO sparse storage
   format. */
cusparseStatus_t  cusparseXcsr2coo(cusparseHandle_t handle,
                                              const int *csrRowPtr,
                                              int nnz,
                                              int m,
                                              int *cooRowInd,
                                              cusparseIndexBase_t idxBase);

/* Description: This routine converts a matrix from CSR to CSC sparse
   storage format. The resulting matrix can be re-interpreted as a
   transpose of the original matrix in CSR storage format. */
cusparseStatus_t  cusparseScsr2csc_v2(cusparseHandle_t handle,
                                                 int m,
                                                 int n,
                                                 int nnz,
                                                 const float  *csrVal,
                                                 const int *csrRowPtr,
                                                 const int *csrColInd,
                                                 float *cscVal,
                                                 int *cscRowInd,
                                                 int *cscColPtr,
                                                 cusparseAction_t copyValues,
                                                 cusparseIndexBase_t idxBase);

cusparseStatus_t  cusparseDcsr2csc_v2(cusparseHandle_t handle,
                                                 int m,
                                                 int n,
                                                 int nnz,
                                                 const double  *csrVal,
                                                 const int *csrRowPtr,
                                                 const int *csrColInd,
                                                 double *cscVal,
                                                 int *cscRowInd,
                                                 int *cscColPtr,
                                                 cusparseAction_t copyValues,
                                                 cusparseIndexBase_t idxBase);

cusparseStatus_t  cusparseCcsr2csc_v2(cusparseHandle_t handle,
                                                 int m,
                                                 int n,
                                                 int nnz,
                                                 const cuComplex  *csrVal,
                                                 const int *csrRowPtr,
                                                 const int *csrColInd,
                                                 cuComplex *cscVal,
                                                 int *cscRowInd,
                                                 int *cscColPtr,
                                                 cusparseAction_t copyValues,
                                                 cusparseIndexBase_t idxBase);

cusparseStatus_t  cusparseZcsr2csc_v2(cusparseHandle_t handle,
                                                 int m,
                                                 int n,
                                                 int nnz,
                                                 const cuDoubleComplex *csrVal,
                                                 const int *csrRowPtr,
                                                 const int *csrColInd,
                                                 cuDoubleComplex *cscVal,
                                                 int *cscRowInd,
                                                 int *cscColPtr,
                                                 cusparseAction_t copyValues,
                                                 cusparseIndexBase_t idxBase);

/* Description: This routine converts a dense matrix to a sparse matrix
   in HYB storage format. */
cusparseStatus_t  cusparseSdense2hyb(cusparseHandle_t handle,
                                                int m,
                                                int n,
                                                const cusparseMatDescr_t descrA,
                                                const float *A,
                                                int lda,
                                                const int *nnzPerRow,
                                                cusparseHybMat_t hybA,
                                                int userEllWidth,
                                                cusparseHybPartition_t partitionType);

cusparseStatus_t  cusparseDdense2hyb(cusparseHandle_t handle,
                                                int m,
                                                int n,
                                                const cusparseMatDescr_t descrA,
                                                const double *A,
                                                int lda,
                                                const int *nnzPerRow,
                                                cusparseHybMat_t hybA,
                                                int userEllWidth,
                                                cusparseHybPartition_t partitionType);

cusparseStatus_t  cusparseCdense2hyb(cusparseHandle_t handle,
                                                int m,
                                                int n,
                                                const cusparseMatDescr_t descrA,
                                                const cuComplex *A,
                                                int lda,
                                                const int *nnzPerRow,
                                                cusparseHybMat_t hybA,
                                                int userEllWidth,
                                                cusparseHybPartition_t partitionType);

cusparseStatus_t  cusparseZdense2hyb(cusparseHandle_t handle,
                                                int m,
                                                int n,
                                                const cusparseMatDescr_t descrA,
                                                const cuDoubleComplex *A,
                                                int lda,
                                                const int *nnzPerRow,
                                                cusparseHybMat_t hybA,
                                                int userEllWidth,
                                                cusparseHybPartition_t partitionType);

/* Description: This routine converts a sparse matrix in HYB storage format
   to a dense matrix. */
cusparseStatus_t  cusparseShyb2dense(cusparseHandle_t handle,
                                                const cusparseMatDescr_t descrA,
                                                const cusparseHybMat_t hybA,
                                                float *A,
                                                int lda);

cusparseStatus_t  cusparseDhyb2dense(cusparseHandle_t handle,
                                                const cusparseMatDescr_t descrA,
                                                const cusparseHybMat_t hybA,
                                                double *A,
                                                int lda);

cusparseStatus_t  cusparseChyb2dense(cusparseHandle_t handle,
                                                const cusparseMatDescr_t descrA,
                                                const cusparseHybMat_t hybA,
                                                cuComplex *A,
                                                int lda);

cusparseStatus_t  cusparseZhyb2dense(cusparseHandle_t handle,
                                                const cusparseMatDescr_t descrA,
                                                const cusparseHybMat_t hybA,
                                                cuDoubleComplex *A,
                                                int lda);

/* Description: This routine converts a sparse matrix in CSR storage format
   to a sparse matrix in HYB storage format. */
cusparseStatus_t  cusparseScsr2hyb(cusparseHandle_t handle,
                                              int m,
                                              int n,
                                              const cusparseMatDescr_t descrA,
                                              const float *csrValA,
                                              const int *csrRowPtrA,
                                              const int *csrColIndA,
                                              cusparseHybMat_t hybA,
                                              int userEllWidth,
                                              cusparseHybPartition_t partitionType);

cusparseStatus_t  cusparseDcsr2hyb(cusparseHandle_t handle,
                                              int m,
                                              int n,
                                              const cusparseMatDescr_t descrA,
                                              const double *csrValA,
                                              const int *csrRowPtrA,
                                              const int *csrColIndA,
                                              cusparseHybMat_t hybA,
                                              int userEllWidth,
                                              cusparseHybPartition_t partitionType);

cusparseStatus_t  cusparseCcsr2hyb(cusparseHandle_t handle,
                                              int m,
                                              int n,
                                              const cusparseMatDescr_t descrA,
                                              const cuComplex *csrValA,
                                              const int *csrRowPtrA,
                                              const int *csrColIndA,
                                              cusparseHybMat_t hybA,
                                              int userEllWidth,
                                              cusparseHybPartition_t partitionType);

cusparseStatus_t  cusparseZcsr2hyb(cusparseHandle_t handle,
                                              int m,
                                              int n,
                                              const cusparseMatDescr_t descrA,
                                              const cuDoubleComplex *csrValA,
                                              const int *csrRowPtrA,
                                              const int *csrColIndA,
                                              cusparseHybMat_t hybA,
                                              int userEllWidth,
                                              cusparseHybPartition_t partitionType);

/* Description: This routine converts a sparse matrix in HYB storage format
   to a sparse matrix in CSR storage format. */
cusparseStatus_t  cusparseShyb2csr(cusparseHandle_t handle,
                                              const cusparseMatDescr_t descrA,
                                              const cusparseHybMat_t hybA,
                                              float *csrValA,
                                              int *csrRowPtrA,
                                              int *csrColIndA);

cusparseStatus_t  cusparseDhyb2csr(cusparseHandle_t handle,
                                              const cusparseMatDescr_t descrA,
                                              const cusparseHybMat_t hybA,
                                              double *csrValA,
                                              int *csrRowPtrA,
                                              int *csrColIndA);

cusparseStatus_t  cusparseChyb2csr(cusparseHandle_t handle,
                                              const cusparseMatDescr_t descrA,
                                              const cusparseHybMat_t hybA,
                                              cuComplex *csrValA,
                                              int *csrRowPtrA,
                                              int *csrColIndA);

cusparseStatus_t  cusparseZhyb2csr(cusparseHandle_t handle,
                                              const cusparseMatDescr_t descrA,
                                              const cusparseHybMat_t hybA,
                                              cuDoubleComplex *csrValA,
                                              int *csrRowPtrA,
                                              int *csrColIndA);

/* Description: This routine converts a sparse matrix in CSC storage format
   to a sparse matrix in HYB storage format. */
cusparseStatus_t  cusparseScsc2hyb(cusparseHandle_t handle,
                                              int m,
                                              int n,
                                              const cusparseMatDescr_t descrA,
                                              const float *cscValA,
                                              const int *cscRowIndA,
                                              const int *cscColPtrA,
                                              cusparseHybMat_t hybA,
                                              int userEllWidth,
                                              cusparseHybPartition_t partitionType);

cusparseStatus_t  cusparseDcsc2hyb(cusparseHandle_t handle,
                                              int m,
                                              int n,
                                              const cusparseMatDescr_t descrA,
                                              const double *cscValA,
                                              const int *cscRowIndA,
                                              const int *cscColPtrA,
                                              cusparseHybMat_t hybA,
                                              int userEllWidth,
                                              cusparseHybPartition_t partitionType);

cusparseStatus_t  cusparseCcsc2hyb(cusparseHandle_t handle,
                                              int m,
                                              int n,
                                              const cusparseMatDescr_t descrA,
                                              const cuComplex *cscValA,
                                              const int *cscRowIndA,
                                              const int *cscColPtrA,
                                              cusparseHybMat_t hybA,
                                              int userEllWidth,
                                              cusparseHybPartition_t partitionType);

cusparseStatus_t  cusparseZcsc2hyb(cusparseHandle_t handle,
                                              int m,
                                              int n,
                                              const cusparseMatDescr_t descrA,
                                              const cuDoubleComplex *cscValA,
                                              const int *cscRowIndA,
                                              const int *cscColPtrA,
                                              cusparseHybMat_t hybA,
                                              int userEllWidth,
                                              cusparseHybPartition_t partitionType);

/* Description: This routine converts a sparse matrix in HYB storage format
   to a sparse matrix in CSC storage format. */
cusparseStatus_t  cusparseShyb2csc(cusparseHandle_t handle,
                                              const cusparseMatDescr_t descrA,
                                              const cusparseHybMat_t hybA,
                                              float *cscVal,
                                              int *cscRowInd,
                                              int *cscColPtr);

cusparseStatus_t  cusparseDhyb2csc(cusparseHandle_t handle,
                                              const cusparseMatDescr_t descrA,
                                              const cusparseHybMat_t hybA,
                                              double *cscVal,
                                              int *cscRowInd,
                                              int *cscColPtr);

cusparseStatus_t  cusparseChyb2csc(cusparseHandle_t handle,
                                              const cusparseMatDescr_t descrA,
                                              const cusparseHybMat_t hybA,
                                              cuComplex *cscVal,
                                              int *cscRowInd,
                                              int *cscColPtr);

cusparseStatus_t  cusparseZhyb2csc(cusparseHandle_t handle,
                                              const cusparseMatDescr_t descrA,
                                              const cusparseHybMat_t hybA,
                                              cuDoubleComplex *cscVal,
                                              int *cscRowInd,
                                              int *cscColPtr);

/* Description: This routine converts a sparse matrix in CSR storage format
   to a sparse matrix in BSR storage format. */
cusparseStatus_t  cusparseXcsr2bsrNnz(cusparseHandle_t handle,
                                                 cusparseDirection_t dirA,
                                                 int m,
                                                 int n,
                                                 const cusparseMatDescr_t descrA,
                                                 const int *csrRowPtrA,
                                                 const int *csrColIndA,
                                                 int blockDim,
                                                 const cusparseMatDescr_t descrC,
                                                 int *bsrRowPtrC,
                                                 int *nnzTotalDevHostPtr);

cusparseStatus_t  cusparseScsr2bsr(cusparseHandle_t handle,
                                              cusparseDirection_t dirA,
                                              int m,
                                              int n,
                                              const cusparseMatDescr_t descrA,
                                              const float *csrValA,
                                              const int *csrRowPtrA,
                                              const int *csrColIndA,
                                              int blockDim,
                                              const cusparseMatDescr_t descrC,
                                              float *bsrValC,
                                              int *bsrRowPtrC,
                                              int *bsrColIndC);

cusparseStatus_t  cusparseDcsr2bsr(cusparseHandle_t handle,
                                              cusparseDirection_t dirA,
                                              int m,
                                              int n,
                                              const cusparseMatDescr_t descrA,
                                              const double *csrValA,
                                              const int *csrRowPtrA,
                                              const int *csrColIndA,
                                              int blockDim,
                                              const cusparseMatDescr_t descrC,
                                              double *bsrValC,
                                              int *bsrRowPtrC,
                                              int *bsrColIndC);

cusparseStatus_t  cusparseCcsr2bsr(cusparseHandle_t handle,
                                              cusparseDirection_t dirA,
                                              int m,
                                              int n,
                                              const cusparseMatDescr_t descrA,
                                              const cuComplex *csrValA,
                                              const int *csrRowPtrA,
                                              const int *csrColIndA,
                                              int blockDim,
                                              const cusparseMatDescr_t descrC,
                                              cuComplex *bsrValC,
                                              int *bsrRowPtrC,
                                              int *bsrColIndC);

cusparseStatus_t  cusparseZcsr2bsr(cusparseHandle_t handle,
                                              cusparseDirection_t dirA,
                                              int m,
                                              int n,
                                              const cusparseMatDescr_t descrA,
                                              const cuDoubleComplex *csrValA,
                                              const int *csrRowPtrA,
                                              const int *csrColIndA,
                                              int blockDim,
                                              const cusparseMatDescr_t descrC,
                                              cuDoubleComplex *bsrValC,
                                              int *bsrRowPtrC,
                                              int *bsrColIndC);

/* Description: This routine converts a sparse matrix in BSR storage format
   to a sparse matrix in CSR storage format. */
cusparseStatus_t  cusparseSbsr2csr(cusparseHandle_t handle,
                                              cusparseDirection_t dirA,
                                              int mb,
                                              int nb,
                                              const cusparseMatDescr_t descrA,
                                              const float *bsrValA,
                                              const int *bsrRowPtrA,
                                              const int *bsrColIndA,
                                              int blockDim,
                                              const cusparseMatDescr_t descrC,
                                              float *csrValC,
                                              int *csrRowPtrC,
                                              int *csrColIndC);

cusparseStatus_t  cusparseDbsr2csr(cusparseHandle_t handle,
                                              cusparseDirection_t dirA,
                                              int mb,
                                              int nb,
                                              const cusparseMatDescr_t descrA,
                                              const double *bsrValA,
                                              const int *bsrRowPtrA,
                                              const int *bsrColIndA,
                                              int   blockDim,
                                              const cusparseMatDescr_t descrC,
                                              double *csrValC,
                                              int *csrRowPtrC,
                                              int *csrColIndC);

cusparseStatus_t  cusparseCbsr2csr(cusparseHandle_t handle,
                                              cusparseDirection_t dirA,
                                              int mb,
                                              int nb,
                                              const cusparseMatDescr_t descrA,
                                              const cuComplex *bsrValA,
                                              const int *bsrRowPtrA,
                                              const int *bsrColIndA,
                                              int blockDim,
                                              const cusparseMatDescr_t descrC,
                                              cuComplex *csrValC,
                                              int *csrRowPtrC,
                                              int *csrColIndC);

cusparseStatus_t  cusparseZbsr2csr(cusparseHandle_t handle,
                                              cusparseDirection_t dirA,
                                              int mb,
                                              int nb,
                                              const cusparseMatDescr_t descrA,
                                              const cuDoubleComplex *bsrValA,
                                              const int *bsrRowPtrA,
                                              const int *bsrColIndA,
                                              int blockDim,
                                              const cusparseMatDescr_t descrC,
                                              cuDoubleComplex *csrValC,
                                              int *csrRowPtrC,
                                              int *csrColIndC);

