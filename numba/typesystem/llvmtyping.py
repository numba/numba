import llvm.core

def get_target_triple():
    target_machine = llvm.ee.TargetMachine.new()
    is_ppc = target_machine.triple.startswith("ppc")
    is_x86 = target_machine.triple.startswith("x86")
    return is_ppc, is_x86


def pointer(base_type):
    if base_type.kind == llvm.core.TYPE_VOID:
        base_type = llvm.core.Type.int(1)
    return llvm.core.Type.pointer(base_type)

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

class Universe:
    def bool(self, context):
        return llvm.core.Type.int(1)

def float(itemsize):
    if itemsize == 4:
        return llvm.core.Type.float()
    elif itemsize == 8:
        return llvm.core.Type.double()
    else:
        is_ppc, is_x86 = get_target_triple()
        if itemsize == 16:
            if is_ppc:
                return llvm.core.Type.ppc_fp128()
            else:
                return llvm.core.Type.fp128()
        else:
            assert itemsize == 10 and is_x86, itemsize
            return llvm.core.Type.x86_fp80()

def struct(type):
    if type.packed:
        lstruct = llvm.core.Type.packed_struct
    else:
        lstruct = llvm.core.Type.struct

    return lstruct([field_type.ty
                        for field_name, field_type in type.fields])

function = llvm.core.Type.function