import llvm.core

def llvm_pointer(ts, type):
    base_type, = type.args
    if base_type.is_void:
        base_type = ts.int_

    llvm_base_type = ts.llvm_type(base_type)
    return llvm.core.Type.pointer(llvm_base_type)

def llvm_float(ts, type):
    if type.itemsize == 4:
        return llvm.core.Type.float()
    elif type.itemsize == 8:
        return lvm.core.Type.double()
    else:
        is_ppc, is_x86 = get_target_triple()
        if self.itemsize == 16:
            if is_ppc:
                return lc.Type.ppc_fp128()
            else:
                return lc.Type.fp128()
        else:
            assert self.itemsize == 10 and is_x86
            return lc.Type.x86_fp80()