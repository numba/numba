import llvm.core

def get_target_triple():
    target_machine = llvm.ee.TargetMachine.new()
    is_ppc = target_machine.triple.startswith("ppc")
    is_x86 = target_machine.triple.startswith("x86")
    return is_ppc, is_x86


def lbool():
    return llvm.core.Type.int(1)

def lint(name, itemsize):
    if name == "bool":
        return lbool()

    return llvm.core.Type.int(itemsize * 8)

def lfloat(name, itemsize):
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

def lstruct(type):
    if type.packed:
        lstruct = llvm.core.Type.packed_struct
    else:
        lstruct = llvm.core.Type.struct

    return lstruct([field_type.ty
                        for field_name, field_type in type.fields])

def lpointer(base_type):
    if base_type.kind == llvm.core.TYPE_VOID:
        base_type = llvm.core.Type.int(1)
    return llvm.core.Type.pointer(base_type)

def lfunction(rettype, argtypes, name=None, is_vararg=False):
    return llvm.core.Type.function(rettype, argtypes, is_vararg)