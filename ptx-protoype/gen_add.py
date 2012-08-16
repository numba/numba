from llvm.core import *
from llvm.ee import *
from llvm.passes import *
from llvm_cbuilder import *
import llvm_cbuilder.shortnames as C

class PTXAdd(CDefinition):
    '''
    Define a 1D kernel for addition of two integer arrays.
    A CUDA kernel must have void return type.
    '''
    _name_ = 'ptx_add'
    _argtys_ = [('S', C.pointer(C.int)),  # sum
                ('A', C.pointer(C.int)),  # left
                ('B', C.pointer(C.int))]  # right

    def body(self, S, A, B):
        tid_x = self.get_intrinsic(INTR_PTX_READ_TID_X, [])
        ntid_x = self.get_intrinsic(INTR_PTX_READ_NTID_X, [])
        ctaid_x = self.get_intrinsic(INTR_PTX_READ_CTAID_X, [])

        tid = self.var_copy(tid_x())
        blkdim = self.var_copy(ntid_x())
        blkid = self.var_copy(ctaid_x())

        i = tid + blkdim * blkid

        S[i].assign(A[i] + B[i])
        self.ret()

def main():
    module = Module.new('mymodule')
    lf_ptxadd = PTXAdd()(module)

    # Must change the calling convention for PTX kernel call
    lf_ptxadd.calling_convention = CC_PTX_KERNEL

    print(module)
    module.verify()

    # Optimize to remove local memory use.
    # Local memory is very slow in CUDA.
    pm = PassManager.new()
    pmb = PassManagerBuilder.new()
    pmb.opt_level = 3
    pmb.populate(pm)

    pm.run(module)

    print('Write PTX')
    ptxtm = TargetMachine.lookup('ptx32', opt=3)

    with open('add.ptx', 'wb') as fout:
        fout.write(ptxtm.emit_assembly(module))


if __name__ == '__main__':
    main()

