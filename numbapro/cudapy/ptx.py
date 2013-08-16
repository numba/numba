'''
This scripts specifies all PTX special objects.
'''
from llvm.core import Type
from numbapro.npm import types, macro

class Stub(object):
    '''A stub object to represent special objects which is meaningless
    outside the context of CUDA-python.
    '''
    _description_ = '<ptx special value>'
    __slots__ = () # don't allocate __dict__
    def __new__(cls):
        raise NotImplementedError("%s is not instantiable" % cls)
    def __repr__(self):
        return self._description_

#-------------------------------------------------------------------------------
# SREG

def _ptx_sreg_tidx(): pass
def _ptx_sreg_tidy(): pass
def _ptx_sreg_tidz(): pass

def _ptx_sreg_ntidx(): pass
def _ptx_sreg_ntidy(): pass
def _ptx_sreg_ntidz(): pass

def _ptx_sreg_ctaidx(): pass
def _ptx_sreg_ctaidy(): pass

def _ptx_sreg_nctaidx(): pass
def _ptx_sreg_nctaidy(): pass


class threadIdx(Stub):
    '''threadIdx.{x, y, z}
    '''
    _description_ = '<threadIdx.{x,y,z}>'

    x = macro.Macro('threadIdx.x', _ptx_sreg_tidx)
    y = macro.Macro('threadIdx.y', _ptx_sreg_tidy)
    z = macro.Macro('threadIdx.z', _ptx_sreg_tidz)

class blockIdx(Stub):
    '''blockIdx.{x, y}
    '''
    _description_ = '<blockIdx.{x,y}>'

    x = macro.Macro('blockIdx.x', _ptx_sreg_ctaidx)
    y = macro.Macro('blockIdx.y', _ptx_sreg_ctaidy)

class blockDim(Stub):
    '''blockDim.{x, y, z}
    '''
    x = macro.Macro('blockDim.x', _ptx_sreg_ntidx)
    y = macro.Macro('blockDim.y', _ptx_sreg_ntidy)
    z = macro.Macro('blockDim.z', _ptx_sreg_ntidz)

class gridDim(Stub):
    '''gridDim.{x, y}
    '''
    _description_ = '<gridDim.{x,y}>'
    x = macro.Macro('gridDim.x', _ptx_sreg_nctaidx)
    y = macro.Macro('gridDim.y', _ptx_sreg_nctaidy)

SREG_MAPPING = {
    _ptx_sreg_tidx: 'llvm.nvvm.read.ptx.sreg.tid.x',
    _ptx_sreg_tidy: 'llvm.nvvm.read.ptx.sreg.tid.y',
    _ptx_sreg_tidz: 'llvm.nvvm.read.ptx.sreg.tid.z',
    
    _ptx_sreg_ntidx: 'llvm.nvvm.read.ptx.sreg.ntid.x',
    _ptx_sreg_ntidy: 'llvm.nvvm.read.ptx.sreg.ntid.y',
    _ptx_sreg_ntidz: 'llvm.nvvm.read.ptx.sreg.ntid.z',
    
    _ptx_sreg_ctaidx: 'llvm.nvvm.read.ptx.sreg.ctaid.x',
    _ptx_sreg_ctaidy: 'llvm.nvvm.read.ptx.sreg.ctaid.y',
    
    _ptx_sreg_nctaidx: 'llvm.nvvm.read.ptx.sreg.nctaid.x',
    _ptx_sreg_nctaidy: 'llvm.nvvm.read.ptx.sreg.nctaid.y',
}

SREG_FUNCTION_TYPE = Type.function(Type.int(), [])
SREG_TYPE = types.uint32

#-------------------------------------------------------------------------------
# Grid Macro

class grid(Stub):
    '''grid(ndim)

    ndim: [uint32] 1 or 2
    
        if ndim == 1:
            return cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
        elif ndim == 2:
            x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
            y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
            return x, y
    '''
    _description_ = '<grid(ndim)>'

#-------------------------------------------------------------------------------
# synthreads

class syncthreads(Stub):
    '''syncthreads() 
    
    Synchronizes all threads in the thread block.
    '''
    _description_ = '<syncthread()>'

#-------------------------------------------------------------------------------
# shared

class shared(Stub):
    '''shared namespace
    '''
    _description_ = '<shared>'

    class array(Stub):
        '''array(shape, dtype)
        
        Allocate shared array of shape and dtype
        '''
        _description_ = '<array>'

#-------------------------------------------------------------------------------
# atomic

class atomic(Stub):
    '''atomic namespace
    '''
    _description_ = '<atomic>'

    class add(Stub):
        '''add(ary, idx, val)
        
        Perform atomic ary[idx] += val
        '''

__all__ = '''
threadIdx
blockIdx
blockDim
gridDim
grid
syncthreads
shared
atomic
'''.split()
