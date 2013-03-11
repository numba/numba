;;; Handwritten IR to access portable atomic instructions
;;; target triple = "x86_64-apple-osx-macho"

define i32 @atomic_add_i32(i32* %ptr, i32 %val){
    %ret = atomicrmw add i32* %ptr, i32 %val seq_cst
    ret i32 %ret
}

define i32 @atomic_xchg_i32(i32* %ptr, i32 %val){
    %ret = atomicrmw xchg i32* %ptr, i32 %val seq_cst
    ret i32 %ret
}

define i32 @atomic_load_i32(i32* %ptr){
    %ret = load atomic i32* %ptr seq_cst, align 4
    ret i32 %ret
}

define void @atomic_store_i32(i32* %ptr, i32 %val){
    store atomic i32 %val, i32* %ptr seq_cst, align 4
    ret void
}

