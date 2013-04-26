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

#func
def to_llvm(self, context):
    assert self.return_type is not None
    self = self.actual_signature
    arg_types = [arg_type.pointer() if arg_type.is_function else arg_type
                     for arg_type in self.args]
    return llvm.core.Type.function(self.return_type.to_llvm(context),
                            [arg_type.to_llvm(context)
                                 for arg_type in arg_types],
                            self.is_vararg)

#carray
def to_llvm(self, context):
    return llvm.core.Type.array(self.base_type.to_llvm(context), self.size)

#int
def to_llvm(self, context):
        if self.itemsize == 1:
            return lc.Type.int(8)
        elif self.itemsize == 2:
            return lc.Type.int(16)
        elif self.itemsize == 4:
            return lc.Type.int(32)
        else:
            assert self.itemsize == 8, self
            return lc.Type.int(64)

    def declare(self):
        if self.name.endswith(('16', '32', '64')):
            return self.name + "_t"
        else:
            return str(self)

class Universe:
    def bool(self, context):
        return llvm.core.Type.int(1)

    def float(self, context):
        if self.itemsize == 4:
            return lc.Type.float()
        elif self.itemsize == 8:
            return lc.Type.double()
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

    def struct(self, context):
        if self.packed:
            lstruct = llvm.core.Type.packed_struct
        else:
            lstruct = llvm.core.Type.struct

        return lstruct([field_type.to_llvm(context)
                           for field_name, field_type in self.fields])
