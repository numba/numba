Example: Sum
------------


.. code-block:: python

    @jit('f8(f8[:])')
    def sum1d(A):
        n = A.shape[0]
        s = 0.0
        for i in range(n):
            s += A[i]
        return s


.. code-block:: LLVM

    "loop_body_6:8":                                  ; preds = %compare.end
      %17 = load i64* %target_temp
      %18 = mul i64 %17, 1
      %19 = add i64 0, %18
      %20 = getelementptr { i64, i32*, i8*, i32, i64*, i64*, i8*, i8*, i32, i8* }* %A, i32 0, i32 2
      %21 = load i8** %20
      %22 = getelementptr { i64, i32*, i8*, i32, i64*, i64*, i8*, i8*, i32, i8* }* %A, i32 0, i32 5
      %23 = load i64** %22
      %24 = getelementptr i64* %23, i32 0
      %25 = load i64* %24, !invariant.load !0
      %26 = mul i64 %19, %25
      %27 = add i64 0, %26
      %28 = getelementptr i8* %21, i64 %27
      %29 = bitcast i8* %28 to double*
      %30 = load double* %29
      %31 = fadd double %s_2, %30
      br label %"for_increment_5:4"


.. code-block:: Assembly

    .section	__TEXT,__text,regular,pure_instructions
        .globl	___numba_specialized___main___2E_sum1d_double_5B__3A__5D_
        .align	4, 0x90
    ___numba_specialized___main___2E_sum1d_double_5B__3A__5D_:
        .cfi_startproc
        pushq	%rbx
    Ltmp2:
        .cfi_def_cfa_offset 16
        subq	$32, %rsp
    Ltmp3:
        .cfi_def_cfa_offset 48
    Ltmp4:
        .cfi_offset %rbx, -16
        movq	%rdi, %rbx
        movq	%rbx, 8(%rsp)
        movabsq	$_Py_IncRef, %rax
        callq	*%rax
        movq	32(%rbx), %rax
        movq	(%rax), %rax
        movq	%rax, 16(%rsp)
        vxorps	%xmm0, %xmm0, %xmm0
        movq	$0, 24(%rsp)
        jmp	LBB0_1
        .align	4, 0x90
    LBB0_5:
        movq	16(%rbx), %rcx
        movq	40(%rbx), %rdx
        movq	24(%rsp), %rax
        movq	(%rdx), %rdx
        imulq	%rax, %rdx
        vaddsd	(%rdx,%rcx), %xmm0, %xmm0
        incl	%eax
        movslq	%eax, %rax
        movq	%rax, 24(%rsp)
    LBB0_1:
        xorb	%al, %al
        movq	24(%rsp), %rcx
        cmpq	16(%rsp), %rcx
        jge	LBB0_2
        movb	$1, %al
    LBB0_2:
        testb	%al, %al
        jne	LBB0_5
        vmovsd	%xmm0, (%rsp)
        movq	8(%rsp), %rdi
        movabsq	$_Py_DecRef, %rax
        callq	*%rax
        vmovsd	(%rsp), %xmm0
        addq	$32, %rsp
        popq	%rbx
        ret
        .cfi_endproc

        .section	__DATA,__datacoal_nt,coalesced
        .globl	_PyArray_API
        .weak_definition	_PyArray_API
        .align	3
    _PyArray_API:
        .quad	4505921120


    .subsections_via_symbols




