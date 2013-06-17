'''
This scripts specifies all PTX special objects.
'''
from llvm.core import Type
from numbapro.npm.types import *

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

class threadIdx(Stub):
    '''threadIdx.{x, y, z}
    '''
    _description_ = '<threadIdx.{x,y,z}>'
    class x(Stub):
        '''threadIdx.x
        '''
        _description_ = '<threadIdx.x>'
    class y(Stub):
        '''threadIdx.y
        '''
        _description_ = '<threadIdx.y>'
    class z(Stub):
        '''threadIdx.z
        '''
        _description_ = '<threadIdx.z>'

class blockIdx(Stub):
    '''blockIdx.{x, y}
    '''
    _description_ = '<blockIdx.{x,y}>'
    class x(Stub):
        '''blockIdx.x
        '''
        _description_ = '<blockIdx.x>'
    class y(Stub):
        '''blockIdx.y
        '''
        _description_ = '<blockIdx.y>'


class blockDim(Stub):
    '''blockDim.{x, y, z}
    '''
    _description_ = '<blockDim.{x,y,z}>'
    class x(Stub):
        '''blockDim.x
        '''
        _description_ = '<blockDim.x>'
    class y(Stub):
        '''blockDim.y
        '''
        _description_ = '<blockDim.y>'
    class z(Stub):
        '''blockDim.z
        '''
        _description_ = '<blockDim.z>'

class gridDim(Stub):
    '''gridDim.{x, y, z}
    '''
    _description_ = '<gridDim.{x,y}>'
    class x(Stub):
        '''gridDim.x
        '''
        _description_ = '<gridDim.x>'
    class y(Stub):
        '''gridDim.y
        '''
        _description_ = '<gridDim.y>'

SREG_MAPPING = {
    threadIdx.x: 'llvm.nvvm.read.ptx.sreg.tid.x',
    threadIdx.y: 'llvm.nvvm.read.ptx.sreg.tid.y',
    threadIdx.z: 'llvm.nvvm.read.ptx.sreg.tid.z',
    
    blockDim.x: 'llvm.nvvm.read.ptx.sreg.ntid.x',
    blockDim.y: 'llvm.nvvm.read.ptx.sreg.ntid.y',
    blockDim.z: 'llvm.nvvm.read.ptx.sreg.ntid.z',
    
    blockIdx.x: 'llvm.nvvm.read.ptx.sreg.ctaid.x',
    blockIdx.y: 'llvm.nvvm.read.ptx.sreg.ctaid.y',
    
    gridDim.x: 'llvm.nvvm.read.ptx.sreg.nctaid.x',
    gridDim.y: 'llvm.nvvm.read.ptx.sreg.nctaid.y',
}

SREG_FUNCTION_TYPE = Type.function(Type.int(), [])
SREG_TYPE = uint32

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


__all__ = '''
threadIdx
blockIdx
blockDim
gridDim
grid
syncthreads
shared
'''.split()
