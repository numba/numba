from ctypes import c_void_p, addressof, c_uint, POINTER, Structure
from numba.cuda.cudadrv.driver import driver, is_device_memory, device_ctypes_pointer, byref
from numba.cuda.compiler import AutoJitCUDAKernel, CUDAKernel

class KernelParams(Structure):
    _fields_ = [
        ('func', c_void_p),
        ('gridDimX', c_uint),
        ('gridDimY', c_uint),
        ('gridDimZ', c_uint),
        ('blockDimX', c_uint),
        ('blockDimY', c_uint),
        ('blockDimZ', c_uint),
        ('sharedMemBytes', c_uint),
        ('kernelParams', POINTER(c_void_p)),
        ('extra', POINTER(c_void_p)),
    ]

class Node:
    pass

class KernelNode(Node):
    def __init__(self, kernel, args = None, deps = None, params = None):
        self.kernel = kernel
        self.args = args or []
        self.deps = deps or []
        self.params = params or { }
        self.builded = { }
        super().__init__()
    
    def _get_params(self):
        params = KernelParams()
        params.blockDimX = self.params.get('blockDimX', 1)
        params.blockDimY = self.params.get('blockDimY', 1)
        params.blockDimZ = self.params.get('blockDimZ', 1)
        params.gridDimX = self.params.get('gridDimX', 1)
        params.gridDimY = self.params.get('gridDimY', 1)
        params.gridDimZ = self.params.get('gridDimZ', 1)
        params.sharedMemBytes = self.params.get('sharedMemBytes', 0)

        if isinstance(self.kernel, AutoJitCUDAKernel):
            kernel = self.kernel.specialize(*self.args)
        elif isinstance(self.kernel, CUDAKernel):
            kernel = self.kernel
        else:
            raise Exception('invalid kernel type "%s"' % type(self.kernel).__name__)
        params.func = kernel._func.get().handle

        retr, kernel_args = [], []
        for t, v in zip(kernel.argument_types, self.args):
            kernel._prepare_args(t, v, 0, retr, kernel_args)
        
        # TODO: take care of retr after graph launched
        if len(retr):
            raise Exception('host array as kernel node args not supported yet')

        param_vals = []
        for arg in kernel_args:
            if is_device_memory(arg):
                param_vals.append(addressof(device_ctypes_pointer(arg)))
            else:
                param_vals.append(addressof(arg))

        params.kernelParams = (c_void_p * len(param_vals))(*param_vals) if len(param_vals) else None
        params.extra = self.params.get('extra', None)
        return params
    
    def build(self, graph = None):
        if not graph:
            graph = Graph()

        if graph in self.builded:
            return self.builded[graph]

        # FIXME: it's strange but we have to build depedencies before other operations
        dep_val = [dep.build(graph).handle for dep in self.deps]
        dep_num = len(dep_val)
        dep_arr = (c_void_p * dep_num)(*dep_val) if dep_num else None

        params = self._get_params()
        builded = self.builded[graph] = BuildedNode(c_void_p(), graph)
        driver.cuGraphAddKernelNode(byref(builded.handle), graph.handle, dep_arr, dep_num, byref(params))
        return builded

class BuildedNode:
    def __init__(self, handle, graph):
        self.handle = handle
        self.graph = graph

    def instantiate(self):
        return GraphExec(self.graph)

    def launch(self, stream = None):
        return GraphExec(self.graph).launch(stream)

class Graph:
    def __init__(self, flags = 0):
        self.handle = c_void_p()
        driver.cuGraphCreate(byref(self.handle), flags)

    def instantiate(self):
        return GraphExec(self)

class GraphExec:
    def __init__(self, graph):
        self.handle = c_void_p()
        driver.cuGraphInstantiate(byref(self.handle), graph.handle, c_void_p(), c_void_p(), 0)
    
    def launch(self, stream = None):
        driver.cuGraphLaunch(self.handle, stream.handle if stream else 0)
