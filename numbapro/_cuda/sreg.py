
# threadIdx

class _threadIdx_x: pass
class _threadIdx_y: pass
class _threadIdx_z: pass

class threadIdx:
    x = _threadIdx_x
    y = _threadIdx_y
    z = _threadIdx_z

# blockIdx

class _blockIdx_x: pass
class _blockIdx_y: pass

class blockIdx:
    x = _blockIdx_x
    y = _blockIdx_y

# blockDim

class _blockDim_x: pass
class _blockDim_y: pass
class _blockDim_z: pass

class blockDim:
    x = _blockDim_x
    y = _blockDim_y
    z = _blockDim_z

# gridDim

class _gridDim_x: pass
class _gridDim_y: pass

class gridDim:
    x = _gridDim_x
    y = _gridDim_y

def _sreg(name):
    def wrap(module):
        fty_sreg =_lc.Type.function(_lc.Type.int(), [])
        return module.get_or_insert_function(fty_sreg, name=name)
    return wrap

SPECIAL_VALUES = {
    _threadIdx_x: 'llvm.nvvm.read.ptx.sreg.tid.x',
    _threadIdx_y: 'llvm.nvvm.read.ptx.sreg.tid.y',
    _threadIdx_z: 'llvm.nvvm.read.ptx.sreg.tid.z',
    
    _blockDim_x: 'llvm.nvvm.read.ptx.sreg.ntid.x',
    _blockDim_y: 'llvm.nvvm.read.ptx.sreg.ntid.y',
    _blockDim_z: 'llvm.nvvm.read.ptx.sreg.ntid.z',
    
    _blockIdx_x: 'llvm.nvvm.read.ptx.sreg.ctaid.x',
    _blockIdx_y: 'llvm.nvvm.read.ptx.sreg.ctaid.y',
    
    _gridDim_x: 'llvm.nvvm.read.ptx.sreg.nctaid.x',
    _gridDim_y: 'llvm.nvvm.read.ptx.sreg.nctaid.y',
}

