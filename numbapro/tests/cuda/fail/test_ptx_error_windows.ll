; ModuleID = 'ptx_matmulcore'

define ptx_kernel void @cuda_wrapper0(i8** nocapture %args, i64** nocapture %dimensions, i64** nocapture %steps, i64* nocapture %arylens, i64 %count) nounwind {
decl:
  %0 = alloca { i64, i32*, i8*, i32, i64*, i64*, i8*, i8*, i32, i8*, i8*, i8*, i64* }
  %1 = alloca { i64, i32*, i8*, i32, i64*, i64*, i8*, i8*, i32, i8*, i8*, i8*, i64* }
  %2 = alloca { i64, i32*, i8*, i32, i64*, i64*, i8*, i8*, i32, i8*, i8*, i8*, i64* }
  %3 = call i32 @llvm.ptx.read.tid.x()
  %4 = call i32 @llvm.ptx.read.ntid.x()
  %5 = call i32 @llvm.ptx.read.ctaid.x()
  %6 = mul i32 %5, %4
  %7 = add i32 %6, %3
  %8 = sext i32 %7 to i64
  %9 = icmp slt i64 %8, %count
  br i1 %9, label %if.end, label %if.then

if.then:                                          ; preds = %BLOCK_156.i.us, %BLOCK_65.i.preheader.lr.ph, %if.end, %decl
  ret void

if.end:                                           ; preds = %decl
  %PyArray.data = getelementptr { i64, i32*, i8*, i32, i64*, i64*, i8*, i8*, i32, i8*, i8*, i8*, i64* }* %0, i32 0, i32 2
  %PyArray.dimensions = getelementptr { i64, i32*, i8*, i32, i64*, i64*, i8*, i8*, i32, i8*, i8*, i8*, i64* }* %0, i32 0, i32 4
  %PyArray.strides = getelementptr { i64, i32*, i8*, i32, i64*, i64*, i8*, i8*, i32, i8*, i8*, i8*, i64* }* %0, i32 0, i32 5
  %PyArray.data8 = getelementptr { i64, i32*, i8*, i32, i64*, i64*, i8*, i8*, i32, i8*, i8*, i8*, i64* }* %1, i32 0, i32 2
  %PyArray.dimensions10 = getelementptr { i64, i32*, i8*, i32, i64*, i64*, i8*, i8*, i32, i8*, i8*, i8*, i64* }* %1, i32 0, i32 4
  %PyArray.strides11 = getelementptr { i64, i32*, i8*, i32, i64*, i64*, i8*, i8*, i32, i8*, i8*, i8*, i64* }* %1, i32 0, i32 5
  %PyArray.data21 = getelementptr { i64, i32*, i8*, i32, i64*, i64*, i8*, i8*, i32, i8*, i8*, i8*, i64* }* %2, i32 0, i32 2
  %PyArray.dimensions23 = getelementptr { i64, i32*, i8*, i32, i64*, i64*, i8*, i8*, i32, i8*, i8*, i8*, i64* }* %2, i32 0, i32 4
  %PyArray.strides24 = getelementptr { i64, i32*, i8*, i32, i64*, i64*, i8*, i8*, i32, i8*, i8*, i8*, i64* }* %2, i32 0, i32 5
  %10 = load i64* %arylens
  %11 = mul i64 %10, %8
  %12 = load i8** %args
  %13 = getelementptr i8* %12, i64 %11
  store i8* %13, i8** %PyArray.data
  %14 = load i64** %dimensions
  store i64* %14, i64** %PyArray.dimensions
  %15 = load i64** %steps
  store i64* %15, i64** %PyArray.strides
  %16 = getelementptr i8** %args, i32 1
  %17 = getelementptr i64* %arylens, i32 1
  %18 = load i64* %17
  %19 = mul i64 %18, %8
  %20 = load i8** %16
  %21 = getelementptr i8* %20, i64 %19
  store i8* %21, i8** %PyArray.data8
  %22 = getelementptr i64** %dimensions, i32 1
  %23 = load i64** %22
  store i64* %23, i64** %PyArray.dimensions10
  %24 = getelementptr i64** %steps, i32 1
  %25 = load i64** %24
  store i64* %25, i64** %PyArray.strides11
  %26 = getelementptr i8** %args, i32 2
  %27 = getelementptr i64* %arylens, i32 2
  %28 = load i64* %27
  %29 = mul i64 %28, %8
  %30 = load i8** %26
  %31 = getelementptr i8* %30, i64 %29
  store i8* %31, i8** %PyArray.data21
  %32 = getelementptr i64** %dimensions, i32 2
  %33 = load i64** %32
  store i64* %33, i64** %PyArray.dimensions23
  %34 = getelementptr i64** %steps, i32 2
  %35 = load i64** %34
  store i64* %35, i64** %PyArray.strides24
  %36 = load i64** %PyArray.dimensions
  %37 = load i64* %36
  %38 = load i64** %PyArray.dimensions10
  %39 = load i64* %38
  %40 = getelementptr i64* %38, i32 1
  %41 = load i64* %40
  %42 = icmp sgt i64 %37, 0
  br i1 %42, label %BLOCK_65.i.preheader.lr.ph, label %if.then

BLOCK_65.i.preheader.lr.ph:                       ; preds = %if.end
  %43 = icmp sgt i64 %41, 0
  %44 = load i8** %PyArray.data21
  %45 = icmp sgt i64 %39, 0
  %46 = load i8** %PyArray.data
  %47 = load i8** %PyArray.data8
  br i1 %43, label %BLOCK_68.i.lr.ph.us, label %if.then

BLOCK_156.i.us:                                   ; preds = %BLOCK_152.i.us.us, %BLOCK_152.i.us32
  %48 = add i64 %56, 1
  %exitcond41 = icmp eq i64 %48, %37
  br i1 %exitcond41, label %if.then, label %BLOCK_68.i.lr.ph.us

BLOCK_152.i.us32:                                 ; preds = %BLOCK_152.i.us32, %BLOCK_68.i.lr.ph.us
  %49 = phi i64 [ %55, %BLOCK_152.i.us32 ], [ 0, %BLOCK_68.i.lr.ph.us ]
  %50 = load i64* %35
  %51 = mul i64 %50, %56
  %52 = getelementptr i8* %44, i64 %51
  %53 = bitcast i8* %52 to float*
  %54 = getelementptr float* %53, i64 %49
  store float 0.000000e+00, float* %54
  %55 = add i64 %49, 1
  %exitcond40 = icmp eq i64 %55, %41
  br i1 %exitcond40, label %BLOCK_156.i.us, label %BLOCK_152.i.us32

BLOCK_68.i.lr.ph.us:                              ; preds = %BLOCK_156.i.us, %BLOCK_65.i.preheader.lr.ph
  %56 = phi i64 [ %48, %BLOCK_156.i.us ], [ 0, %BLOCK_65.i.preheader.lr.ph ]
  br i1 %45, label %BLOCK_103.i.lr.ph.us.us, label %BLOCK_152.i.us32

BLOCK_152.i.us.us:                                ; preds = %BLOCK_103.i.us.us
  %57 = add i64 %58, 1
  %exitcond39 = icmp eq i64 %57, %41
  br i1 %exitcond39, label %BLOCK_156.i.us, label %BLOCK_103.i.lr.ph.us.us

BLOCK_103.i.lr.ph.us.us:                          ; preds = %BLOCK_68.i.lr.ph.us, %BLOCK_152.i.us.us
  %58 = phi i64 [ %57, %BLOCK_152.i.us.us ], [ 0, %BLOCK_68.i.lr.ph.us ]
  %59 = load i64* %35
  %60 = mul i64 %59, %56
  %61 = getelementptr i8* %44, i64 %60
  %62 = bitcast i8* %61 to float*
  %63 = getelementptr float* %62, i64 %58
  store float 0.000000e+00, float* %63
  br label %BLOCK_103.i.us.us

BLOCK_103.i.us.us:                                ; preds = %BLOCK_103.i.us.us, %BLOCK_103.i.lr.ph.us.us
  %64 = phi i64 [ 0, %BLOCK_103.i.lr.ph.us.us ], [ %85, %BLOCK_103.i.us.us ]
  %65 = load i64* %35
  %66 = mul i64 %65, %56
  %67 = getelementptr i8* %44, i64 %66
  %68 = bitcast i8* %67 to float*
  %69 = getelementptr float* %68, i64 %58
  %70 = load float* %69
  %71 = load i64* %15
  %72 = mul i64 %71, %56
  %73 = getelementptr i8* %46, i64 %72
  %74 = bitcast i8* %73 to float*
  %75 = getelementptr float* %74, i64 %64
  %76 = load float* %75
  %77 = load i64* %25
  %78 = mul i64 %77, %64
  %79 = getelementptr i8* %47, i64 %78
  %80 = bitcast i8* %79 to float*
  %81 = getelementptr float* %80, i64 %58
  %82 = load float* %81
  %83 = fmul float %76, %82
  %84 = fadd float %70, %83
  store float %84, float* %69
  %85 = add i64 %64, 1
  %exitcond = icmp eq i64 %85, %39
  br i1 %exitcond, label %BLOCK_152.i.us.us, label %BLOCK_103.i.us.us
}

declare i32 @llvm.ptx.read.tid.x() nounwind readnone

declare i32 @llvm.ptx.read.ntid.x() nounwind readnone

declare i32 @llvm.ptx.read.ctaid.x() nounwind readnone
