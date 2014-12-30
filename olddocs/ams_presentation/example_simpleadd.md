# Example: Sum

## Python

```python
    @jit(f8(f8[:]))
    def sum1d(A):
        n = A.shape[0]
        s = 0.0
        for i in range(n):
            s += A[i]
        return s

```

# Example: Sum

## LLVM IR

```LLVM
"loop_body_6:8":
    ...
    %24 = getelementptr i64* %23, i32 0
    %25 = load i64* %24, !invariant.load !0
    %26 = mul i64 %19, %25
    %27 = add i64 0, %26
    %28 = getelementptr i8* %21, i64 %27
    %29 = bitcast i8* %28 to double*
    %30 = load double* %29
    %31 = fadd double %s_2, %30
    br label %"for_increment_5:4"

```

# Example: Sum

## x86 Assembly


```Assembly
    ...
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
```

