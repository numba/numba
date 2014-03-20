# This file is generated automatically from tools/parseheaders.py
# Do not modified the content of this file

cusparseCreate = ('cusparseStatus_t', (('handle', 'cusparseHandle_t*'),))


cusparseDestroy = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'),))


cusparseGetVersion = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('version', 'int*'),))


cusparseSetStream = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('streamId', 'cudaStream_t'),))


cusparseGetPointerMode = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('mode', 'cusparsePointerMode_t*'),))


cusparseSetPointerMode = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('mode', 'cusparsePointerMode_t'),))


cusparseCreateMatDescr = ('cusparseStatus_t', (('descrA', 'cusparseMatDescr_t*'),))


cusparseDestroyMatDescr = ('cusparseStatus_t', (('descrA', 'cusparseMatDescr_t'),))


cusparseSetMatType = ('cusparseStatus_t', (('descrA', 'cusparseMatDescr_t'), ('type', 'cusparseMatrixType_t'),))


cusparseGetMatType = ('cusparseMatrixType_t', (('descrA', 'cusparseMatDescr_t'),))


cusparseSetMatFillMode = ('cusparseStatus_t', (('descrA', 'cusparseMatDescr_t'), ('fillMode', 'cusparseFillMode_t'),))


cusparseGetMatFillMode = ('cusparseFillMode_t', (('descrA', 'cusparseMatDescr_t'),))


cusparseSetMatDiagType = ('cusparseStatus_t', (('descrA', 'cusparseMatDescr_t'), ('diagType', 'cusparseDiagType_t'),))


cusparseGetMatDiagType = ('cusparseDiagType_t', (('descrA', 'cusparseMatDescr_t'),))


cusparseSetMatIndexBase = ('cusparseStatus_t', (('descrA', 'cusparseMatDescr_t'), ('base', 'cusparseIndexBase_t'),))


cusparseGetMatIndexBase = ('cusparseIndexBase_t', (('descrA', 'cusparseMatDescr_t'),))


cusparseCreateSolveAnalysisInfo = ('cusparseStatus_t', (('info', 'cusparseSolveAnalysisInfo_t*'),))


cusparseDestroySolveAnalysisInfo = ('cusparseStatus_t', (('info', 'cusparseSolveAnalysisInfo_t'),))


cusparseGetLevelInfo = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('info', 'cusparseSolveAnalysisInfo_t'), ('nlevels', 'int*'), ('levelPtr', 'int**'), ('levelInd', 'int**'),))


cusparseCreateHybMat = ('cusparseStatus_t', (('hybA', 'cusparseHybMat_t*'),))


cusparseDestroyHybMat = ('cusparseStatus_t', (('hybA', 'cusparseHybMat_t'),))


cusparseSaxpyi_v2 = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('nnz', 'int'), ('alpha', 'float*'), ('xVal', 'float*'), ('xInd', 'int*'), ('y', 'float*'), ('idxBase', 'cusparseIndexBase_t'),))


cusparseDaxpyi_v2 = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('nnz', 'int'), ('alpha', 'double*'), ('xVal', 'double*'), ('xInd', 'int*'), ('y', 'double*'), ('idxBase', 'cusparseIndexBase_t'),))


cusparseCaxpyi_v2 = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('nnz', 'int'), ('alpha', 'cuComplex*'), ('xVal', 'cuComplex*'), ('xInd', 'int*'), ('y', 'cuComplex*'), ('idxBase', 'cusparseIndexBase_t'),))


cusparseZaxpyi_v2 = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('nnz', 'int'), ('alpha', 'cuDoubleComplex*'), ('xVal', 'cuDoubleComplex*'), ('xInd', 'int*'), ('y', 'cuDoubleComplex*'), ('idxBase', 'cusparseIndexBase_t'),))


cusparseSdoti = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('nnz', 'int'), ('xVal', 'float*'), ('xInd', 'int*'), ('y', 'float*'), ('resultDevHostPtr', 'float*'), ('idxBase', 'cusparseIndexBase_t'),))


cusparseDdoti = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('nnz', 'int'), ('xVal', 'double*'), ('xInd', 'int*'), ('y', 'double*'), ('resultDevHostPtr', 'double*'), ('idxBase', 'cusparseIndexBase_t'),))


cusparseCdoti = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('nnz', 'int'), ('xVal', 'cuComplex*'), ('xInd', 'int*'), ('y', 'cuComplex*'), ('resultDevHostPtr', 'cuComplex*'), ('idxBase', 'cusparseIndexBase_t'),))


cusparseZdoti = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('nnz', 'int'), ('xVal', 'cuDoubleComplex*'), ('xInd', 'int*'), ('y', 'cuDoubleComplex*'), ('resultDevHostPtr', 'cuDoubleComplex*'), ('idxBase', 'cusparseIndexBase_t'),))


cusparseCdotci = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('nnz', 'int'), ('xVal', 'cuComplex*'), ('xInd', 'int*'), ('y', 'cuComplex*'), ('resultDevHostPtr', 'cuComplex*'), ('idxBase', 'cusparseIndexBase_t'),))


cusparseZdotci = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('nnz', 'int'), ('xVal', 'cuDoubleComplex*'), ('xInd', 'int*'), ('y', 'cuDoubleComplex*'), ('resultDevHostPtr', 'cuDoubleComplex*'), ('idxBase', 'cusparseIndexBase_t'),))


cusparseSgthr = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('nnz', 'int'), ('y', 'float*'), ('xVal', 'float*'), ('xInd', 'int*'), ('idxBase', 'cusparseIndexBase_t'),))


cusparseDgthr = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('nnz', 'int'), ('y', 'double*'), ('xVal', 'double*'), ('xInd', 'int*'), ('idxBase', 'cusparseIndexBase_t'),))


cusparseCgthr = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('nnz', 'int'), ('y', 'cuComplex*'), ('xVal', 'cuComplex*'), ('xInd', 'int*'), ('idxBase', 'cusparseIndexBase_t'),))


cusparseZgthr = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('nnz', 'int'), ('y', 'cuDoubleComplex*'), ('xVal', 'cuDoubleComplex*'), ('xInd', 'int*'), ('idxBase', 'cusparseIndexBase_t'),))


cusparseSgthrz = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('nnz', 'int'), ('y', 'float*'), ('xVal', 'float*'), ('xInd', 'int*'), ('idxBase', 'cusparseIndexBase_t'),))


cusparseDgthrz = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('nnz', 'int'), ('y', 'double*'), ('xVal', 'double*'), ('xInd', 'int*'), ('idxBase', 'cusparseIndexBase_t'),))


cusparseCgthrz = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('nnz', 'int'), ('y', 'cuComplex*'), ('xVal', 'cuComplex*'), ('xInd', 'int*'), ('idxBase', 'cusparseIndexBase_t'),))


cusparseZgthrz = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('nnz', 'int'), ('y', 'cuDoubleComplex*'), ('xVal', 'cuDoubleComplex*'), ('xInd', 'int*'), ('idxBase', 'cusparseIndexBase_t'),))


cusparseSsctr = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('nnz', 'int'), ('xVal', 'float*'), ('xInd', 'int*'), ('y', 'float*'), ('idxBase', 'cusparseIndexBase_t'),))


cusparseDsctr = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('nnz', 'int'), ('xVal', 'double*'), ('xInd', 'int*'), ('y', 'double*'), ('idxBase', 'cusparseIndexBase_t'),))


cusparseCsctr = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('nnz', 'int'), ('xVal', 'cuComplex*'), ('xInd', 'int*'), ('y', 'cuComplex*'), ('idxBase', 'cusparseIndexBase_t'),))


cusparseZsctr = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('nnz', 'int'), ('xVal', 'cuDoubleComplex*'), ('xInd', 'int*'), ('y', 'cuDoubleComplex*'), ('idxBase', 'cusparseIndexBase_t'),))


cusparseSroti_v2 = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('nnz', 'int'), ('xVal', 'float*'), ('xInd', 'int*'), ('y', 'float*'), ('c', 'float*'), ('s', 'float*'), ('idxBase', 'cusparseIndexBase_t'),))


cusparseDroti_v2 = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('nnz', 'int'), ('xVal', 'double*'), ('xInd', 'int*'), ('y', 'double*'), ('c', 'double*'), ('s', 'double*'), ('idxBase', 'cusparseIndexBase_t'),))


cusparseScsrmv_v2 = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('transA', 'cusparseOperation_t'), ('m', 'int'), ('n', 'int'), ('nnz', 'int'), ('alpha', 'float*'), ('descrA', 'cusparseMatDescr_t'), ('csrValA', 'float*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('x', 'float*'), ('beta', 'float*'), ('y', 'float*'),))


cusparseDcsrmv_v2 = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('transA', 'cusparseOperation_t'), ('m', 'int'), ('n', 'int'), ('nnz', 'int'), ('alpha', 'double*'), ('descrA', 'cusparseMatDescr_t'), ('csrValA', 'double*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('x', 'double*'), ('beta', 'double*'), ('y', 'double*'),))


cusparseCcsrmv_v2 = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('transA', 'cusparseOperation_t'), ('m', 'int'), ('n', 'int'), ('nnz', 'int'), ('alpha', 'cuComplex*'), ('descrA', 'cusparseMatDescr_t'), ('csrValA', 'cuComplex*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('x', 'cuComplex*'), ('beta', 'cuComplex*'), ('y', 'cuComplex*'),))


cusparseZcsrmv_v2 = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('transA', 'cusparseOperation_t'), ('m', 'int'), ('n', 'int'), ('nnz', 'int'), ('alpha', 'cuDoubleComplex*'), ('descrA', 'cusparseMatDescr_t'), ('csrValA', 'cuDoubleComplex*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('x', 'cuDoubleComplex*'), ('beta', 'cuDoubleComplex*'), ('y', 'cuDoubleComplex*'),))


cusparseShybmv = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('transA', 'cusparseOperation_t'), ('alpha', 'float*'), ('descrA', 'cusparseMatDescr_t'), ('hybA', 'cusparseHybMat_t'), ('x', 'float*'), ('beta', 'float*'), ('y', 'float*'),))


cusparseDhybmv = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('transA', 'cusparseOperation_t'), ('alpha', 'double*'), ('descrA', 'cusparseMatDescr_t'), ('hybA', 'cusparseHybMat_t'), ('x', 'double*'), ('beta', 'double*'), ('y', 'double*'),))


cusparseChybmv = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('transA', 'cusparseOperation_t'), ('alpha', 'cuComplex*'), ('descrA', 'cusparseMatDescr_t'), ('hybA', 'cusparseHybMat_t'), ('x', 'cuComplex*'), ('beta', 'cuComplex*'), ('y', 'cuComplex*'),))


cusparseZhybmv = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('transA', 'cusparseOperation_t'), ('alpha', 'cuDoubleComplex*'), ('descrA', 'cusparseMatDescr_t'), ('hybA', 'cusparseHybMat_t'), ('x', 'cuDoubleComplex*'), ('beta', 'cuDoubleComplex*'), ('y', 'cuDoubleComplex*'),))


cusparseSbsrmv = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('dirA', 'cusparseDirection_t'), ('transA', 'cusparseOperation_t'), ('mb', 'int'), ('nb', 'int'), ('nnzb', 'int'), ('alpha', 'float*'), ('descrA', 'cusparseMatDescr_t'), ('bsrValA', 'float*'), ('bsrRowPtrA', 'int*'), ('bsrColIndA', 'int*'), ('blockDim', 'int'), ('x', 'float*'), ('beta', 'float*'), ('y', 'float*'),))


cusparseDbsrmv = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('dirA', 'cusparseDirection_t'), ('transA', 'cusparseOperation_t'), ('mb', 'int'), ('nb', 'int'), ('nnzb', 'int'), ('alpha', 'double*'), ('descrA', 'cusparseMatDescr_t'), ('bsrValA', 'double*'), ('bsrRowPtrA', 'int*'), ('bsrColIndA', 'int*'), ('blockDim', 'int'), ('x', 'double*'), ('beta', 'double*'), ('y', 'double*'),))


cusparseCbsrmv = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('dirA', 'cusparseDirection_t'), ('transA', 'cusparseOperation_t'), ('mb', 'int'), ('nb', 'int'), ('nnzb', 'int'), ('alpha', 'cuComplex*'), ('descrA', 'cusparseMatDescr_t'), ('bsrValA', 'cuComplex*'), ('bsrRowPtrA', 'int*'), ('bsrColIndA', 'int*'), ('blockDim', 'int'), ('x', 'cuComplex*'), ('beta', 'cuComplex*'), ('y', 'cuComplex*'),))


cusparseZbsrmv = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('dirA', 'cusparseDirection_t'), ('transA', 'cusparseOperation_t'), ('mb', 'int'), ('nb', 'int'), ('nnzb', 'int'), ('alpha', 'cuDoubleComplex*'), ('descrA', 'cusparseMatDescr_t'), ('bsrValA', 'cuDoubleComplex*'), ('bsrRowPtrA', 'int*'), ('bsrColIndA', 'int*'), ('blockDim', 'int'), ('x', 'cuDoubleComplex*'), ('beta', 'cuDoubleComplex*'), ('y', 'cuDoubleComplex*'),))


cusparseSbsrxmv = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('dirA', 'cusparseDirection_t'), ('transA', 'cusparseOperation_t'), ('sizeOfMask', 'int'), ('mb', 'int'), ('nb', 'int'), ('nnzb', 'int'), ('alpha', 'float*'), ('descrA', 'cusparseMatDescr_t'), ('bsrValA', 'float*'), ('bsrMaskPtrA', 'int*'), ('bsrRowPtrA', 'int*'), ('bsrEndPtrA', 'int*'), ('bsrColIndA', 'int*'), ('blockDim', 'int'), ('x', 'float*'), ('beta', 'float*'), ('y', 'float*'),))


cusparseDbsrxmv = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('dirA', 'cusparseDirection_t'), ('transA', 'cusparseOperation_t'), ('sizeOfMask', 'int'), ('mb', 'int'), ('nb', 'int'), ('nnzb', 'int'), ('alpha', 'double*'), ('descrA', 'cusparseMatDescr_t'), ('bsrValA', 'double*'), ('bsrMaskPtrA', 'int*'), ('bsrRowPtrA', 'int*'), ('bsrEndPtrA', 'int*'), ('bsrColIndA', 'int*'), ('blockDim', 'int'), ('x', 'double*'), ('beta', 'double*'), ('y', 'double*'),))


cusparseCbsrxmv = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('dirA', 'cusparseDirection_t'), ('transA', 'cusparseOperation_t'), ('sizeOfMask', 'int'), ('mb', 'int'), ('nb', 'int'), ('nnzb', 'int'), ('alpha', 'cuComplex*'), ('descrA', 'cusparseMatDescr_t'), ('bsrValA', 'cuComplex*'), ('bsrMaskPtrA', 'int*'), ('bsrRowPtrA', 'int*'), ('bsrEndPtrA', 'int*'), ('bsrColIndA', 'int*'), ('blockDim', 'int'), ('x', 'cuComplex*'), ('beta', 'cuComplex*'), ('y', 'cuComplex*'),))


cusparseZbsrxmv = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('dirA', 'cusparseDirection_t'), ('transA', 'cusparseOperation_t'), ('sizeOfMask', 'int'), ('mb', 'int'), ('nb', 'int'), ('nnzb', 'int'), ('alpha', 'cuDoubleComplex*'), ('descrA', 'cusparseMatDescr_t'), ('bsrValA', 'cuDoubleComplex*'), ('bsrMaskPtrA', 'int*'), ('bsrRowPtrA', 'int*'), ('bsrEndPtrA', 'int*'), ('bsrColIndA', 'int*'), ('blockDim', 'int'), ('x', 'cuDoubleComplex*'), ('beta', 'cuDoubleComplex*'), ('y', 'cuDoubleComplex*'),))


cusparseScsrsv_analysis_v2 = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('transA', 'cusparseOperation_t'), ('m', 'int'), ('nnz', 'int'), ('descrA', 'cusparseMatDescr_t'), ('csrValA', 'float*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('info', 'cusparseSolveAnalysisInfo_t'),))


cusparseDcsrsv_analysis_v2 = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('transA', 'cusparseOperation_t'), ('m', 'int'), ('nnz', 'int'), ('descrA', 'cusparseMatDescr_t'), ('csrValA', 'double*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('info', 'cusparseSolveAnalysisInfo_t'),))


cusparseCcsrsv_analysis_v2 = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('transA', 'cusparseOperation_t'), ('m', 'int'), ('nnz', 'int'), ('descrA', 'cusparseMatDescr_t'), ('csrValA', 'cuComplex*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('info', 'cusparseSolveAnalysisInfo_t'),))


cusparseZcsrsv_analysis_v2 = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('transA', 'cusparseOperation_t'), ('m', 'int'), ('nnz', 'int'), ('descrA', 'cusparseMatDescr_t'), ('csrValA', 'cuDoubleComplex*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('info', 'cusparseSolveAnalysisInfo_t'),))


cusparseScsrsv_solve_v2 = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('transA', 'cusparseOperation_t'), ('m', 'int'), ('alpha', 'float*'), ('descrA', 'cusparseMatDescr_t'), ('csrValA', 'float*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('info', 'cusparseSolveAnalysisInfo_t'), ('x', 'float*'), ('y', 'float*'),))


cusparseDcsrsv_solve_v2 = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('transA', 'cusparseOperation_t'), ('m', 'int'), ('alpha', 'double*'), ('descrA', 'cusparseMatDescr_t'), ('csrValA', 'double*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('info', 'cusparseSolveAnalysisInfo_t'), ('x', 'double*'), ('y', 'double*'),))


cusparseCcsrsv_solve_v2 = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('transA', 'cusparseOperation_t'), ('m', 'int'), ('alpha', 'cuComplex*'), ('descrA', 'cusparseMatDescr_t'), ('csrValA', 'cuComplex*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('info', 'cusparseSolveAnalysisInfo_t'), ('x', 'cuComplex*'), ('y', 'cuComplex*'),))


cusparseZcsrsv_solve_v2 = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('transA', 'cusparseOperation_t'), ('m', 'int'), ('alpha', 'cuDoubleComplex*'), ('descrA', 'cusparseMatDescr_t'), ('csrValA', 'cuDoubleComplex*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('info', 'cusparseSolveAnalysisInfo_t'), ('x', 'cuDoubleComplex*'), ('y', 'cuDoubleComplex*'),))


cusparseShybsv_analysis = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('transA', 'cusparseOperation_t'), ('descrA', 'cusparseMatDescr_t'), ('hybA', 'cusparseHybMat_t'), ('info', 'cusparseSolveAnalysisInfo_t'),))


cusparseDhybsv_analysis = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('transA', 'cusparseOperation_t'), ('descrA', 'cusparseMatDescr_t'), ('hybA', 'cusparseHybMat_t'), ('info', 'cusparseSolveAnalysisInfo_t'),))


cusparseChybsv_analysis = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('transA', 'cusparseOperation_t'), ('descrA', 'cusparseMatDescr_t'), ('hybA', 'cusparseHybMat_t'), ('info', 'cusparseSolveAnalysisInfo_t'),))


cusparseZhybsv_analysis = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('transA', 'cusparseOperation_t'), ('descrA', 'cusparseMatDescr_t'), ('hybA', 'cusparseHybMat_t'), ('info', 'cusparseSolveAnalysisInfo_t'),))


cusparseShybsv_solve = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('trans', 'cusparseOperation_t'), ('alpha', 'float*'), ('descra', 'cusparseMatDescr_t'), ('hybA', 'cusparseHybMat_t'), ('info', 'cusparseSolveAnalysisInfo_t'), ('x', 'float*'), ('y', 'float*'),))


cusparseChybsv_solve = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('trans', 'cusparseOperation_t'), ('alpha', 'cuComplex*'), ('descra', 'cusparseMatDescr_t'), ('hybA', 'cusparseHybMat_t'), ('info', 'cusparseSolveAnalysisInfo_t'), ('x', 'cuComplex*'), ('y', 'cuComplex*'),))


cusparseDhybsv_solve = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('trans', 'cusparseOperation_t'), ('alpha', 'double*'), ('descra', 'cusparseMatDescr_t'), ('hybA', 'cusparseHybMat_t'), ('info', 'cusparseSolveAnalysisInfo_t'), ('x', 'double*'), ('y', 'double*'),))


cusparseZhybsv_solve = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('trans', 'cusparseOperation_t'), ('alpha', 'cuDoubleComplex*'), ('descra', 'cusparseMatDescr_t'), ('hybA', 'cusparseHybMat_t'), ('info', 'cusparseSolveAnalysisInfo_t'), ('x', 'cuDoubleComplex*'), ('y', 'cuDoubleComplex*'),))


cusparseScsrmm_v2 = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('transA', 'cusparseOperation_t'), ('m', 'int'), ('n', 'int'), ('k', 'int'), ('nnz', 'int'), ('alpha', 'float*'), ('descrA', 'cusparseMatDescr_t'), ('csrValA', 'float*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('B', 'float*'), ('ldb', 'int'), ('beta', 'float*'), ('C', 'float*'), ('ldc', 'int'),))


cusparseDcsrmm_v2 = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('transA', 'cusparseOperation_t'), ('m', 'int'), ('n', 'int'), ('k', 'int'), ('nnz', 'int'), ('alpha', 'double*'), ('descrA', 'cusparseMatDescr_t'), ('csrValA', 'double*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('B', 'double*'), ('ldb', 'int'), ('beta', 'double*'), ('C', 'double*'), ('ldc', 'int'),))


cusparseCcsrmm_v2 = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('transA', 'cusparseOperation_t'), ('m', 'int'), ('n', 'int'), ('k', 'int'), ('nnz', 'int'), ('alpha', 'cuComplex*'), ('descrA', 'cusparseMatDescr_t'), ('csrValA', 'cuComplex*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('B', 'cuComplex*'), ('ldb', 'int'), ('beta', 'cuComplex*'), ('C', 'cuComplex*'), ('ldc', 'int'),))


cusparseZcsrmm_v2 = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('transA', 'cusparseOperation_t'), ('m', 'int'), ('n', 'int'), ('k', 'int'), ('nnz', 'int'), ('alpha', 'cuDoubleComplex*'), ('descrA', 'cusparseMatDescr_t'), ('csrValA', 'cuDoubleComplex*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('B', 'cuDoubleComplex*'), ('ldb', 'int'), ('beta', 'cuDoubleComplex*'), ('C', 'cuDoubleComplex*'), ('ldc', 'int'),))


cusparseScsrmm2 = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('transa', 'cusparseOperation_t'), ('transb', 'cusparseOperation_t'), ('m', 'int'), ('n', 'int'), ('k', 'int'), ('nnz', 'int'), ('alpha', 'float*'), ('descrA', 'cusparseMatDescr_t'), ('csrValA', 'float*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('B', 'float*'), ('ldb', 'int'), ('beta', 'float*'), ('C', 'float*'), ('ldc', 'int'),))


cusparseDcsrmm2 = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('transa', 'cusparseOperation_t'), ('transb', 'cusparseOperation_t'), ('m', 'int'), ('n', 'int'), ('k', 'int'), ('nnz', 'int'), ('alpha', 'double*'), ('descrA', 'cusparseMatDescr_t'), ('csrValA', 'double*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('B', 'double*'), ('ldb', 'int'), ('beta', 'double*'), ('C', 'double*'), ('ldc', 'int'),))


cusparseCcsrmm2 = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('transa', 'cusparseOperation_t'), ('transb', 'cusparseOperation_t'), ('m', 'int'), ('n', 'int'), ('k', 'int'), ('nnz', 'int'), ('alpha', 'cuComplex*'), ('descrA', 'cusparseMatDescr_t'), ('csrValA', 'cuComplex*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('B', 'cuComplex*'), ('ldb', 'int'), ('beta', 'cuComplex*'), ('C', 'cuComplex*'), ('ldc', 'int'),))


cusparseZcsrmm2 = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('transa', 'cusparseOperation_t'), ('transb', 'cusparseOperation_t'), ('m', 'int'), ('n', 'int'), ('k', 'int'), ('nnz', 'int'), ('alpha', 'cuDoubleComplex*'), ('descrA', 'cusparseMatDescr_t'), ('csrValA', 'cuDoubleComplex*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('B', 'cuDoubleComplex*'), ('ldb', 'int'), ('beta', 'cuDoubleComplex*'), ('C', 'cuDoubleComplex*'), ('ldc', 'int'),))


cusparseScsrsm_analysis = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('transA', 'cusparseOperation_t'), ('m', 'int'), ('nnz', 'int'), ('descrA', 'cusparseMatDescr_t'), ('csrValA', 'float*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('info', 'cusparseSolveAnalysisInfo_t'),))


cusparseDcsrsm_analysis = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('transA', 'cusparseOperation_t'), ('m', 'int'), ('nnz', 'int'), ('descrA', 'cusparseMatDescr_t'), ('csrValA', 'double*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('info', 'cusparseSolveAnalysisInfo_t'),))


cusparseCcsrsm_analysis = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('transA', 'cusparseOperation_t'), ('m', 'int'), ('nnz', 'int'), ('descrA', 'cusparseMatDescr_t'), ('csrValA', 'cuComplex*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('info', 'cusparseSolveAnalysisInfo_t'),))


cusparseZcsrsm_analysis = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('transA', 'cusparseOperation_t'), ('m', 'int'), ('nnz', 'int'), ('descrA', 'cusparseMatDescr_t'), ('csrValA', 'cuDoubleComplex*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('info', 'cusparseSolveAnalysisInfo_t'),))


cusparseScsrsm_solve = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('transA', 'cusparseOperation_t'), ('m', 'int'), ('n', 'int'), ('alpha', 'float*'), ('descrA', 'cusparseMatDescr_t'), ('csrValA', 'float*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('info', 'cusparseSolveAnalysisInfo_t'), ('x', 'float*'), ('ldx', 'int'), ('y', 'float*'), ('ldy', 'int'),))


cusparseDcsrsm_solve = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('transA', 'cusparseOperation_t'), ('m', 'int'), ('n', 'int'), ('alpha', 'double*'), ('descrA', 'cusparseMatDescr_t'), ('csrValA', 'double*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('info', 'cusparseSolveAnalysisInfo_t'), ('x', 'double*'), ('ldx', 'int'), ('y', 'double*'), ('ldy', 'int'),))


cusparseCcsrsm_solve = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('transA', 'cusparseOperation_t'), ('m', 'int'), ('n', 'int'), ('alpha', 'cuComplex*'), ('descrA', 'cusparseMatDescr_t'), ('csrValA', 'cuComplex*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('info', 'cusparseSolveAnalysisInfo_t'), ('x', 'cuComplex*'), ('ldx', 'int'), ('y', 'cuComplex*'), ('ldy', 'int'),))


cusparseZcsrsm_solve = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('transA', 'cusparseOperation_t'), ('m', 'int'), ('n', 'int'), ('alpha', 'cuDoubleComplex*'), ('descrA', 'cusparseMatDescr_t'), ('csrValA', 'cuDoubleComplex*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('info', 'cusparseSolveAnalysisInfo_t'), ('x', 'cuDoubleComplex*'), ('ldx', 'int'), ('y', 'cuDoubleComplex*'), ('ldy', 'int'),))


cusparseScsrilu0 = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('trans', 'cusparseOperation_t'), ('m', 'int'), ('descrA', 'cusparseMatDescr_t'), ('csrValA_ValM', 'float*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('info', 'cusparseSolveAnalysisInfo_t'),))


cusparseDcsrilu0 = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('trans', 'cusparseOperation_t'), ('m', 'int'), ('descrA', 'cusparseMatDescr_t'), ('csrValA_ValM', 'double*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('info', 'cusparseSolveAnalysisInfo_t'),))


cusparseCcsrilu0 = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('trans', 'cusparseOperation_t'), ('m', 'int'), ('descrA', 'cusparseMatDescr_t'), ('csrValA_ValM', 'cuComplex*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('info', 'cusparseSolveAnalysisInfo_t'),))


cusparseZcsrilu0 = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('trans', 'cusparseOperation_t'), ('m', 'int'), ('descrA', 'cusparseMatDescr_t'), ('csrValA_ValM', 'cuDoubleComplex*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('info', 'cusparseSolveAnalysisInfo_t'),))


cusparseScsric0 = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('trans', 'cusparseOperation_t'), ('m', 'int'), ('descrA', 'cusparseMatDescr_t'), ('csrValA_ValM', 'float*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('info', 'cusparseSolveAnalysisInfo_t'),))


cusparseDcsric0 = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('trans', 'cusparseOperation_t'), ('m', 'int'), ('descrA', 'cusparseMatDescr_t'), ('csrValA_ValM', 'double*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('info', 'cusparseSolveAnalysisInfo_t'),))


cusparseCcsric0 = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('trans', 'cusparseOperation_t'), ('m', 'int'), ('descrA', 'cusparseMatDescr_t'), ('csrValA_ValM', 'cuComplex*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('info', 'cusparseSolveAnalysisInfo_t'),))


cusparseZcsric0 = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('trans', 'cusparseOperation_t'), ('m', 'int'), ('descrA', 'cusparseMatDescr_t'), ('csrValA_ValM', 'cuDoubleComplex*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('info', 'cusparseSolveAnalysisInfo_t'),))


cusparseSgtsv = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('dl', 'float*'), ('d', 'float*'), ('du', 'float*'), ('B', 'float*'), ('ldb', 'int'),))


cusparseDgtsv = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('dl', 'double*'), ('d', 'double*'), ('du', 'double*'), ('B', 'double*'), ('ldb', 'int'),))


cusparseCgtsv = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('dl', 'cuComplex*'), ('d', 'cuComplex*'), ('du', 'cuComplex*'), ('B', 'cuComplex*'), ('ldb', 'int'),))


cusparseZgtsv = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('dl', 'cuDoubleComplex*'), ('d', 'cuDoubleComplex*'), ('du', 'cuDoubleComplex*'), ('B', 'cuDoubleComplex*'), ('ldb', 'int'),))


cusparseSgtsv_nopivot = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('dl', 'float*'), ('d', 'float*'), ('du', 'float*'), ('B', 'float*'), ('ldb', 'int'),))


cusparseDgtsv_nopivot = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('dl', 'double*'), ('d', 'double*'), ('du', 'double*'), ('B', 'double*'), ('ldb', 'int'),))


cusparseCgtsv_nopivot = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('dl', 'cuComplex*'), ('d', 'cuComplex*'), ('du', 'cuComplex*'), ('B', 'cuComplex*'), ('ldb', 'int'),))


cusparseZgtsv_nopivot = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('dl', 'cuDoubleComplex*'), ('d', 'cuDoubleComplex*'), ('du', 'cuDoubleComplex*'), ('B', 'cuDoubleComplex*'), ('ldb', 'int'),))


cusparseSgtsvStridedBatch = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('dl', 'float*'), ('d', 'float*'), ('du', 'float*'), ('x', 'float*'), ('batchCount', 'int'), ('batchStride', 'int'),))


cusparseDgtsvStridedBatch = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('dl', 'double*'), ('d', 'double*'), ('du', 'double*'), ('x', 'double*'), ('batchCount', 'int'), ('batchStride', 'int'),))


cusparseCgtsvStridedBatch = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('dl', 'cuComplex*'), ('d', 'cuComplex*'), ('du', 'cuComplex*'), ('x', 'cuComplex*'), ('batchCount', 'int'), ('batchStride', 'int'),))


cusparseZgtsvStridedBatch = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('dl', 'cuDoubleComplex*'), ('d', 'cuDoubleComplex*'), ('du', 'cuDoubleComplex*'), ('x', 'cuDoubleComplex*'), ('batchCount', 'int'), ('batchStride', 'int'),))


cusparseXcsrgemmNnz = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('transA', 'cusparseOperation_t'), ('transB', 'cusparseOperation_t'), ('m', 'int'), ('n', 'int'), ('k', 'int'), ('descrA', 'cusparseMatDescr_t'), ('nnzA', 'int'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('descrB', 'cusparseMatDescr_t'), ('nnzB', 'int'), ('csrRowPtrB', 'int*'), ('csrColIndB', 'int*'), ('descrC', 'cusparseMatDescr_t'), ('csrRowPtrC', 'int*'), ('nnzTotalDevHostPtr', 'int*'),))


cusparseScsrgemm = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('transA', 'cusparseOperation_t'), ('transB', 'cusparseOperation_t'), ('m', 'int'), ('n', 'int'), ('k', 'int'), ('descrA', 'cusparseMatDescr_t'), ('nnzA', 'int'), ('csrValA', 'float*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('descrB', 'cusparseMatDescr_t'), ('nnzB', 'int'), ('csrValB', 'float*'), ('csrRowPtrB', 'int*'), ('csrColIndB', 'int*'), ('descrC', 'cusparseMatDescr_t'), ('csrValC', 'float*'), ('csrRowPtrC', 'int*'), ('csrColIndC', 'int*'),))


cusparseDcsrgemm = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('transA', 'cusparseOperation_t'), ('transB', 'cusparseOperation_t'), ('m', 'int'), ('n', 'int'), ('k', 'int'), ('descrA', 'cusparseMatDescr_t'), ('nnzA', 'int'), ('csrValA', 'double*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('descrB', 'cusparseMatDescr_t'), ('nnzB', 'int'), ('csrValB', 'double*'), ('csrRowPtrB', 'int*'), ('csrColIndB', 'int*'), ('descrC', 'cusparseMatDescr_t'), ('csrValC', 'double*'), ('csrRowPtrC', 'int*'), ('csrColIndC', 'int*'),))


cusparseCcsrgemm = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('transA', 'cusparseOperation_t'), ('transB', 'cusparseOperation_t'), ('m', 'int'), ('n', 'int'), ('k', 'int'), ('descrA', 'cusparseMatDescr_t'), ('nnzA', 'int'), ('csrValA', 'cuComplex*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('descrB', 'cusparseMatDescr_t'), ('nnzB', 'int'), ('csrValB', 'cuComplex*'), ('csrRowPtrB', 'int*'), ('csrColIndB', 'int*'), ('descrC', 'cusparseMatDescr_t'), ('csrValC', 'cuComplex*'), ('csrRowPtrC', 'int*'), ('csrColIndC', 'int*'),))


cusparseZcsrgemm = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('transA', 'cusparseOperation_t'), ('transB', 'cusparseOperation_t'), ('m', 'int'), ('n', 'int'), ('k', 'int'), ('descrA', 'cusparseMatDescr_t'), ('nnzA', 'int'), ('csrValA', 'cuDoubleComplex*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('descrB', 'cusparseMatDescr_t'), ('nnzB', 'int'), ('csrValB', 'cuDoubleComplex*'), ('csrRowPtrB', 'int*'), ('csrColIndB', 'int*'), ('descrC', 'cusparseMatDescr_t'), ('csrValC', 'cuDoubleComplex*'), ('csrRowPtrC', 'int*'), ('csrColIndC', 'int*'),))


cusparseXcsrgeamNnz = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('descrA', 'cusparseMatDescr_t'), ('nnzA', 'int'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('descrB', 'cusparseMatDescr_t'), ('nnzB', 'int'), ('csrRowPtrB', 'int*'), ('csrColIndB', 'int*'), ('descrC', 'cusparseMatDescr_t'), ('csrRowPtrC', 'int*'), ('nnzTotalDevHostPtr', 'int*'),))


cusparseScsrgeam = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('alpha', 'float*'), ('descrA', 'cusparseMatDescr_t'), ('nnzA', 'int'), ('csrValA', 'float*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('beta', 'float*'), ('descrB', 'cusparseMatDescr_t'), ('nnzB', 'int'), ('csrValB', 'float*'), ('csrRowPtrB', 'int*'), ('csrColIndB', 'int*'), ('descrC', 'cusparseMatDescr_t'), ('csrValC', 'float*'), ('csrRowPtrC', 'int*'), ('csrColIndC', 'int*'),))


cusparseDcsrgeam = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('alpha', 'double*'), ('descrA', 'cusparseMatDescr_t'), ('nnzA', 'int'), ('csrValA', 'double*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('beta', 'double*'), ('descrB', 'cusparseMatDescr_t'), ('nnzB', 'int'), ('csrValB', 'double*'), ('csrRowPtrB', 'int*'), ('csrColIndB', 'int*'), ('descrC', 'cusparseMatDescr_t'), ('csrValC', 'double*'), ('csrRowPtrC', 'int*'), ('csrColIndC', 'int*'),))


cusparseCcsrgeam = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('alpha', 'cuComplex*'), ('descrA', 'cusparseMatDescr_t'), ('nnzA', 'int'), ('csrValA', 'cuComplex*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('beta', 'cuComplex*'), ('descrB', 'cusparseMatDescr_t'), ('nnzB', 'int'), ('csrValB', 'cuComplex*'), ('csrRowPtrB', 'int*'), ('csrColIndB', 'int*'), ('descrC', 'cusparseMatDescr_t'), ('csrValC', 'cuComplex*'), ('csrRowPtrC', 'int*'), ('csrColIndC', 'int*'),))


cusparseZcsrgeam = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('alpha', 'cuDoubleComplex*'), ('descrA', 'cusparseMatDescr_t'), ('nnzA', 'int'), ('csrValA', 'cuDoubleComplex*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('beta', 'cuDoubleComplex*'), ('descrB', 'cusparseMatDescr_t'), ('nnzB', 'int'), ('csrValB', 'cuDoubleComplex*'), ('csrRowPtrB', 'int*'), ('csrColIndB', 'int*'), ('descrC', 'cusparseMatDescr_t'), ('csrValC', 'cuDoubleComplex*'), ('csrRowPtrC', 'int*'), ('csrColIndC', 'int*'),))


cusparseSnnz = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('dirA', 'cusparseDirection_t'), ('m', 'int'), ('n', 'int'), ('descrA', 'cusparseMatDescr_t'), ('A', 'float*'), ('lda', 'int'), ('nnzPerRowCol', 'int*'), ('nnzTotalDevHostPtr', 'int*'),))


cusparseDnnz = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('dirA', 'cusparseDirection_t'), ('m', 'int'), ('n', 'int'), ('descrA', 'cusparseMatDescr_t'), ('A', 'double*'), ('lda', 'int'), ('nnzPerRowCol', 'int*'), ('nnzTotalDevHostPtr', 'int*'),))


cusparseCnnz = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('dirA', 'cusparseDirection_t'), ('m', 'int'), ('n', 'int'), ('descrA', 'cusparseMatDescr_t'), ('A', 'cuComplex*'), ('lda', 'int'), ('nnzPerRowCol', 'int*'), ('nnzTotalDevHostPtr', 'int*'),))


cusparseZnnz = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('dirA', 'cusparseDirection_t'), ('m', 'int'), ('n', 'int'), ('descrA', 'cusparseMatDescr_t'), ('A', 'cuDoubleComplex*'), ('lda', 'int'), ('nnzPerRowCol', 'int*'), ('nnzTotalDevHostPtr', 'int*'),))


cusparseSdense2csr = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('descrA', 'cusparseMatDescr_t'), ('A', 'float*'), ('lda', 'int'), ('nnzPerRow', 'int*'), ('csrValA', 'float*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'),))


cusparseDdense2csr = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('descrA', 'cusparseMatDescr_t'), ('A', 'double*'), ('lda', 'int'), ('nnzPerRow', 'int*'), ('csrValA', 'double*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'),))


cusparseCdense2csr = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('descrA', 'cusparseMatDescr_t'), ('A', 'cuComplex*'), ('lda', 'int'), ('nnzPerRow', 'int*'), ('csrValA', 'cuComplex*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'),))


cusparseZdense2csr = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('descrA', 'cusparseMatDescr_t'), ('A', 'cuDoubleComplex*'), ('lda', 'int'), ('nnzPerRow', 'int*'), ('csrValA', 'cuDoubleComplex*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'),))


cusparseScsr2dense = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('descrA', 'cusparseMatDescr_t'), ('csrValA', 'float*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('A', 'float*'), ('lda', 'int'),))


cusparseDcsr2dense = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('descrA', 'cusparseMatDescr_t'), ('csrValA', 'double*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('A', 'double*'), ('lda', 'int'),))


cusparseCcsr2dense = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('descrA', 'cusparseMatDescr_t'), ('csrValA', 'cuComplex*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('A', 'cuComplex*'), ('lda', 'int'),))


cusparseZcsr2dense = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('descrA', 'cusparseMatDescr_t'), ('csrValA', 'cuDoubleComplex*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('A', 'cuDoubleComplex*'), ('lda', 'int'),))


cusparseSdense2csc = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('descrA', 'cusparseMatDescr_t'), ('A', 'float*'), ('lda', 'int'), ('nnzPerCol', 'int*'), ('cscValA', 'float*'), ('cscRowIndA', 'int*'), ('cscColPtrA', 'int*'),))


cusparseDdense2csc = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('descrA', 'cusparseMatDescr_t'), ('A', 'double*'), ('lda', 'int'), ('nnzPerCol', 'int*'), ('cscValA', 'double*'), ('cscRowIndA', 'int*'), ('cscColPtrA', 'int*'),))


cusparseCdense2csc = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('descrA', 'cusparseMatDescr_t'), ('A', 'cuComplex*'), ('lda', 'int'), ('nnzPerCol', 'int*'), ('cscValA', 'cuComplex*'), ('cscRowIndA', 'int*'), ('cscColPtrA', 'int*'),))


cusparseZdense2csc = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('descrA', 'cusparseMatDescr_t'), ('A', 'cuDoubleComplex*'), ('lda', 'int'), ('nnzPerCol', 'int*'), ('cscValA', 'cuDoubleComplex*'), ('cscRowIndA', 'int*'), ('cscColPtrA', 'int*'),))


cusparseScsc2dense = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('descrA', 'cusparseMatDescr_t'), ('cscValA', 'float*'), ('cscRowIndA', 'int*'), ('cscColPtrA', 'int*'), ('A', 'float*'), ('lda', 'int'),))


cusparseDcsc2dense = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('descrA', 'cusparseMatDescr_t'), ('cscValA', 'double*'), ('cscRowIndA', 'int*'), ('cscColPtrA', 'int*'), ('A', 'double*'), ('lda', 'int'),))


cusparseCcsc2dense = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('descrA', 'cusparseMatDescr_t'), ('cscValA', 'cuComplex*'), ('cscRowIndA', 'int*'), ('cscColPtrA', 'int*'), ('A', 'cuComplex*'), ('lda', 'int'),))


cusparseZcsc2dense = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('descrA', 'cusparseMatDescr_t'), ('cscValA', 'cuDoubleComplex*'), ('cscRowIndA', 'int*'), ('cscColPtrA', 'int*'), ('A', 'cuDoubleComplex*'), ('lda', 'int'),))


cusparseXcoo2csr = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('cooRowInd', 'int*'), ('nnz', 'int'), ('m', 'int'), ('csrRowPtr', 'int*'), ('idxBase', 'cusparseIndexBase_t'),))


cusparseXcsr2coo = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('csrRowPtr', 'int*'), ('nnz', 'int'), ('m', 'int'), ('cooRowInd', 'int*'), ('idxBase', 'cusparseIndexBase_t'),))


cusparseScsr2csc_v2 = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('nnz', 'int'), ('csrVal', 'float*'), ('csrRowPtr', 'int*'), ('csrColInd', 'int*'), ('cscVal', 'float*'), ('cscRowInd', 'int*'), ('cscColPtr', 'int*'), ('copyValues', 'cusparseAction_t'), ('idxBase', 'cusparseIndexBase_t'),))


cusparseDcsr2csc_v2 = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('nnz', 'int'), ('csrVal', 'double*'), ('csrRowPtr', 'int*'), ('csrColInd', 'int*'), ('cscVal', 'double*'), ('cscRowInd', 'int*'), ('cscColPtr', 'int*'), ('copyValues', 'cusparseAction_t'), ('idxBase', 'cusparseIndexBase_t'),))


cusparseCcsr2csc_v2 = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('nnz', 'int'), ('csrVal', 'cuComplex*'), ('csrRowPtr', 'int*'), ('csrColInd', 'int*'), ('cscVal', 'cuComplex*'), ('cscRowInd', 'int*'), ('cscColPtr', 'int*'), ('copyValues', 'cusparseAction_t'), ('idxBase', 'cusparseIndexBase_t'),))


cusparseZcsr2csc_v2 = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('nnz', 'int'), ('csrVal', 'cuDoubleComplex*'), ('csrRowPtr', 'int*'), ('csrColInd', 'int*'), ('cscVal', 'cuDoubleComplex*'), ('cscRowInd', 'int*'), ('cscColPtr', 'int*'), ('copyValues', 'cusparseAction_t'), ('idxBase', 'cusparseIndexBase_t'),))


cusparseSdense2hyb = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('descrA', 'cusparseMatDescr_t'), ('A', 'float*'), ('lda', 'int'), ('nnzPerRow', 'int*'), ('hybA', 'cusparseHybMat_t'), ('userEllWidth', 'int'), ('partitionType', 'cusparseHybPartition_t'),))


cusparseDdense2hyb = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('descrA', 'cusparseMatDescr_t'), ('A', 'double*'), ('lda', 'int'), ('nnzPerRow', 'int*'), ('hybA', 'cusparseHybMat_t'), ('userEllWidth', 'int'), ('partitionType', 'cusparseHybPartition_t'),))


cusparseCdense2hyb = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('descrA', 'cusparseMatDescr_t'), ('A', 'cuComplex*'), ('lda', 'int'), ('nnzPerRow', 'int*'), ('hybA', 'cusparseHybMat_t'), ('userEllWidth', 'int'), ('partitionType', 'cusparseHybPartition_t'),))


cusparseZdense2hyb = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('descrA', 'cusparseMatDescr_t'), ('A', 'cuDoubleComplex*'), ('lda', 'int'), ('nnzPerRow', 'int*'), ('hybA', 'cusparseHybMat_t'), ('userEllWidth', 'int'), ('partitionType', 'cusparseHybPartition_t'),))


cusparseShyb2dense = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('descrA', 'cusparseMatDescr_t'), ('hybA', 'cusparseHybMat_t'), ('A', 'float*'), ('lda', 'int'),))


cusparseDhyb2dense = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('descrA', 'cusparseMatDescr_t'), ('hybA', 'cusparseHybMat_t'), ('A', 'double*'), ('lda', 'int'),))


cusparseChyb2dense = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('descrA', 'cusparseMatDescr_t'), ('hybA', 'cusparseHybMat_t'), ('A', 'cuComplex*'), ('lda', 'int'),))


cusparseZhyb2dense = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('descrA', 'cusparseMatDescr_t'), ('hybA', 'cusparseHybMat_t'), ('A', 'cuDoubleComplex*'), ('lda', 'int'),))


cusparseScsr2hyb = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('descrA', 'cusparseMatDescr_t'), ('csrValA', 'float*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('hybA', 'cusparseHybMat_t'), ('userEllWidth', 'int'), ('partitionType', 'cusparseHybPartition_t'),))


cusparseDcsr2hyb = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('descrA', 'cusparseMatDescr_t'), ('csrValA', 'double*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('hybA', 'cusparseHybMat_t'), ('userEllWidth', 'int'), ('partitionType', 'cusparseHybPartition_t'),))


cusparseCcsr2hyb = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('descrA', 'cusparseMatDescr_t'), ('csrValA', 'cuComplex*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('hybA', 'cusparseHybMat_t'), ('userEllWidth', 'int'), ('partitionType', 'cusparseHybPartition_t'),))


cusparseZcsr2hyb = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('descrA', 'cusparseMatDescr_t'), ('csrValA', 'cuDoubleComplex*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('hybA', 'cusparseHybMat_t'), ('userEllWidth', 'int'), ('partitionType', 'cusparseHybPartition_t'),))


cusparseShyb2csr = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('descrA', 'cusparseMatDescr_t'), ('hybA', 'cusparseHybMat_t'), ('csrValA', 'float*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'),))


cusparseDhyb2csr = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('descrA', 'cusparseMatDescr_t'), ('hybA', 'cusparseHybMat_t'), ('csrValA', 'double*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'),))


cusparseChyb2csr = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('descrA', 'cusparseMatDescr_t'), ('hybA', 'cusparseHybMat_t'), ('csrValA', 'cuComplex*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'),))


cusparseZhyb2csr = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('descrA', 'cusparseMatDescr_t'), ('hybA', 'cusparseHybMat_t'), ('csrValA', 'cuDoubleComplex*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'),))


cusparseScsc2hyb = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('descrA', 'cusparseMatDescr_t'), ('cscValA', 'float*'), ('cscRowIndA', 'int*'), ('cscColPtrA', 'int*'), ('hybA', 'cusparseHybMat_t'), ('userEllWidth', 'int'), ('partitionType', 'cusparseHybPartition_t'),))


cusparseDcsc2hyb = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('descrA', 'cusparseMatDescr_t'), ('cscValA', 'double*'), ('cscRowIndA', 'int*'), ('cscColPtrA', 'int*'), ('hybA', 'cusparseHybMat_t'), ('userEllWidth', 'int'), ('partitionType', 'cusparseHybPartition_t'),))


cusparseCcsc2hyb = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('descrA', 'cusparseMatDescr_t'), ('cscValA', 'cuComplex*'), ('cscRowIndA', 'int*'), ('cscColPtrA', 'int*'), ('hybA', 'cusparseHybMat_t'), ('userEllWidth', 'int'), ('partitionType', 'cusparseHybPartition_t'),))


cusparseZcsc2hyb = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('m', 'int'), ('n', 'int'), ('descrA', 'cusparseMatDescr_t'), ('cscValA', 'cuDoubleComplex*'), ('cscRowIndA', 'int*'), ('cscColPtrA', 'int*'), ('hybA', 'cusparseHybMat_t'), ('userEllWidth', 'int'), ('partitionType', 'cusparseHybPartition_t'),))


cusparseShyb2csc = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('descrA', 'cusparseMatDescr_t'), ('hybA', 'cusparseHybMat_t'), ('cscVal', 'float*'), ('cscRowInd', 'int*'), ('cscColPtr', 'int*'),))


cusparseDhyb2csc = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('descrA', 'cusparseMatDescr_t'), ('hybA', 'cusparseHybMat_t'), ('cscVal', 'double*'), ('cscRowInd', 'int*'), ('cscColPtr', 'int*'),))


cusparseChyb2csc = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('descrA', 'cusparseMatDescr_t'), ('hybA', 'cusparseHybMat_t'), ('cscVal', 'cuComplex*'), ('cscRowInd', 'int*'), ('cscColPtr', 'int*'),))


cusparseZhyb2csc = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('descrA', 'cusparseMatDescr_t'), ('hybA', 'cusparseHybMat_t'), ('cscVal', 'cuDoubleComplex*'), ('cscRowInd', 'int*'), ('cscColPtr', 'int*'),))


cusparseXcsr2bsrNnz = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('dirA', 'cusparseDirection_t'), ('m', 'int'), ('n', 'int'), ('descrA', 'cusparseMatDescr_t'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('blockDim', 'int'), ('descrC', 'cusparseMatDescr_t'), ('bsrRowPtrC', 'int*'), ('nnzTotalDevHostPtr', 'int*'),))


cusparseScsr2bsr = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('dirA', 'cusparseDirection_t'), ('m', 'int'), ('n', 'int'), ('descrA', 'cusparseMatDescr_t'), ('csrValA', 'float*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('blockDim', 'int'), ('descrC', 'cusparseMatDescr_t'), ('bsrValC', 'float*'), ('bsrRowPtrC', 'int*'), ('bsrColIndC', 'int*'),))


cusparseDcsr2bsr = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('dirA', 'cusparseDirection_t'), ('m', 'int'), ('n', 'int'), ('descrA', 'cusparseMatDescr_t'), ('csrValA', 'double*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('blockDim', 'int'), ('descrC', 'cusparseMatDescr_t'), ('bsrValC', 'double*'), ('bsrRowPtrC', 'int*'), ('bsrColIndC', 'int*'),))


cusparseCcsr2bsr = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('dirA', 'cusparseDirection_t'), ('m', 'int'), ('n', 'int'), ('descrA', 'cusparseMatDescr_t'), ('csrValA', 'cuComplex*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('blockDim', 'int'), ('descrC', 'cusparseMatDescr_t'), ('bsrValC', 'cuComplex*'), ('bsrRowPtrC', 'int*'), ('bsrColIndC', 'int*'),))


cusparseZcsr2bsr = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('dirA', 'cusparseDirection_t'), ('m', 'int'), ('n', 'int'), ('descrA', 'cusparseMatDescr_t'), ('csrValA', 'cuDoubleComplex*'), ('csrRowPtrA', 'int*'), ('csrColIndA', 'int*'), ('blockDim', 'int'), ('descrC', 'cusparseMatDescr_t'), ('bsrValC', 'cuDoubleComplex*'), ('bsrRowPtrC', 'int*'), ('bsrColIndC', 'int*'),))


cusparseSbsr2csr = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('dirA', 'cusparseDirection_t'), ('mb', 'int'), ('nb', 'int'), ('descrA', 'cusparseMatDescr_t'), ('bsrValA', 'float*'), ('bsrRowPtrA', 'int*'), ('bsrColIndA', 'int*'), ('blockDim', 'int'), ('descrC', 'cusparseMatDescr_t'), ('csrValC', 'float*'), ('csrRowPtrC', 'int*'), ('csrColIndC', 'int*'),))


cusparseDbsr2csr = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('dirA', 'cusparseDirection_t'), ('mb', 'int'), ('nb', 'int'), ('descrA', 'cusparseMatDescr_t'), ('bsrValA', 'double*'), ('bsrRowPtrA', 'int*'), ('bsrColIndA', 'int*'), ('blockDim', 'int'), ('descrC', 'cusparseMatDescr_t'), ('csrValC', 'double*'), ('csrRowPtrC', 'int*'), ('csrColIndC', 'int*'),))


cusparseCbsr2csr = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('dirA', 'cusparseDirection_t'), ('mb', 'int'), ('nb', 'int'), ('descrA', 'cusparseMatDescr_t'), ('bsrValA', 'cuComplex*'), ('bsrRowPtrA', 'int*'), ('bsrColIndA', 'int*'), ('blockDim', 'int'), ('descrC', 'cusparseMatDescr_t'), ('csrValC', 'cuComplex*'), ('csrRowPtrC', 'int*'), ('csrColIndC', 'int*'),))


cusparseZbsr2csr = ('cusparseStatus_t', (('handle', 'cusparseHandle_t'), ('dirA', 'cusparseDirection_t'), ('mb', 'int'), ('nb', 'int'), ('descrA', 'cusparseMatDescr_t'), ('bsrValA', 'cuDoubleComplex*'), ('bsrRowPtrA', 'int*'), ('bsrColIndA', 'int*'), ('blockDim', 'int'), ('descrC', 'cusparseMatDescr_t'), ('csrValC', 'cuDoubleComplex*'), ('csrRowPtrC', 'int*'), ('csrColIndC', 'int*'),))

