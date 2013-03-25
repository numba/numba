import ast

from llvm.core import Constant, Type

from numba.codegen import translate as _translate
from numba.type_inference import infer as _infer
from numba import nodes, llvm_types
from numba.minivect import minitypes

from .special_values import sreg, smem, barrier, macros
from .nvvm import ADDRSPACE_SHARED

from .nodes import (CudaAttributeNode,
                    CudaSMemArrayNode,
                    CudaSMemArrayCallNode,
                    CudaSMemAssignNode,
                    CudaMacroGridNode,
                    CudaMacroGridExpandValuesNode)

class CudaTypeInferer(_infer.TypeInferer):
    "Override type-inferer to handle cuda specific attributes"
    def __init__(self, *args, **kws):
        super(CudaTypeInferer, self).__init__(*args, **kws)
#        self.return_variable = None

    def visit_Attribute(self, node):
        from numbapro import cuda as _THIS_MODULE

        value = self.visit(node.value)
        retval = node # default to return the original node

        if isinstance(node.value, ast.Name):
            #assert isinstance(value.ctx, ast.Load)
            obj = self.func.func_globals.get(node.value.id)
            if obj is _THIS_MODULE:
                retval = CudaAttributeNode(_THIS_MODULE).resolve(node.attr)
        elif isinstance(value, CudaAttributeNode):
            retval = value.resolve(node.attr)

        if retval.value in sreg.SPECIAL_VALUES:  # sreg
            # subsitute with a function call
            sig = minitypes.FunctionType(minitypes.uint32, [])
            fname = sreg.SPECIAL_VALUES[retval.value]
            funcnode = nodes.LLVMExternalFunctionNode(sig, fname)
            callnode = nodes.NativeFunctionCallNode(sig, funcnode, [])
            retval = callnode
        elif retval.value == smem._array:   # allocate shared memory
            retval = CudaSMemArrayNode()
        elif retval.value == barrier.syncthreads: # syncthreads
            sig = minitypes.FunctionType(minitypes.void, [])
            fname = 'llvm.nvvm.barrier0'
            funcnode = nodes.LLVMExternalFunctionNode(sig, fname)
            retval = funcnode
        elif retval.value == macros.grid:  # expand into sreg attributes
            retval = CudaMacroGridNode()
        if retval is node:
            retval = super(CudaTypeInferer, self).visit_Attribute(node)

        return retval

    def visit_Call(self, node):
        from .decorators import CudaDeviceFunction
        func = self.visit(node.func)
        if isinstance(func, CudaSMemArrayNode):
            assert len(node.args) <= 2
            kws = dict((kw.arg, kw.value)for kw in node.keywords)

            arglist = 'shape', 'dtype'
            for i, v in enumerate(node.args):
                k = arglist[i]
                if k in kws:
                    raise KeyError("%s is re-defined as keyword argument" % k)
                else:
                    kws[k] = v

            shape = tuple()

            arg_shape = kws['shape']
            if hasattr(arg_shape, 'elts'):
                for elem in arg_shape.elts:
                    node = self.visit(elem)
                    shape += (node.pyval,)
            else:
                shape = (arg_shape.n,)

            dtype_id = kws['dtype'].id # FIXME must be a ast.Name
            dtype = self.func.func_globals[dtype_id] # FIXME must be a Numba type

            node = CudaSMemArrayCallNode(self.context, shape=shape, dtype=dtype)
            return node
        elif isinstance(func, nodes.LLVMExternalFunctionNode):
            self.visitlist(node.args)
            self.visitlist(node.keywords)
            callnode = nodes.NativeFunctionCallNode(func.signature, func,
                                                    node.args)
            return callnode
        elif isinstance(func, CudaMacroGridNode):
            assert len(node.args) == 1
            assert len(node.keywords) == 0
            ndim = self.visit(node.args[0]).pyval
            if ndim == 1:
                node = _make_sreg_pattern(sreg.threadIdx.x,
                                          sreg.blockIdx.x,
                                          sreg.blockDim.x)
                return self.visit(node)
            elif ndim == 2:
                node1 = _make_sreg_pattern(sreg.threadIdx.x,
                                           sreg.blockIdx.x,
                                           sreg.blockDim.x)
                node2 = _make_sreg_pattern(sreg.threadIdx.y,
                                           sreg.blockIdx.y,
                                           sreg.blockDim.y)

                return self.visit(ast.Tuple(elts=[node1, node2],
                                            ctx=ast.Load()))
            else:
                raise ValueError("Dimension is only valid for 1 or 2, " \
                                 "but got %d" % ndim)
        elif (func.variable.is_constant and
              isinstance(func.variable.constant_value, CudaDeviceFunction)):
            devicefunc = func.variable.constant_value
            self.visitlist(node.args)
            func = nodes.LLVMExternalFunctionNode(devicefunc.signature,
                                                  devicefunc.lfunc.name)
            callnode = nodes.NativeFunctionCallNode(devicefunc.signature,
                                                    func, node.args)
            return callnode
        else:
            return super(CudaTypeInferer, self).visit_Call(node)

    def visit_Assign(self, node):
        node.inplace_op = getattr(node, 'inplace_op', None)
        node.value = self.visit(node.value)

        if isinstance(node.value, CudaSMemArrayCallNode):
            errmsg = "LHS of shared memory declaration can have only one value."
            assert len(node.targets) == 1, errmsg
            target = node.targets[0] = self.visit(node.targets[0])
            node = CudaSMemAssignNode(node.targets[0], node.value)
            self.assign(target, node.value)
            return node

        # FIXME: the following is copied from TypeInferer.visit_Assign
        #        there seems to be some side-effect in visit(node.value)
        if len(node.targets) != 1 or isinstance(node.targets[0], (ast.List,
                                                                  ast.Tuple)):
            return self._handle_unpacking(node)

        target = node.targets[0] = self.visit(node.targets[0])
        self.assign(target, node.value)

        lhs_var = target.variable
        rhs_var = node.value.variable
        if isinstance(target, ast.Name):
            node.value = nodes.CoercionNode(node.value, lhs_var.type)
        elif lhs_var.type != rhs_var.type:
            if lhs_var.type.is_array and rhs_var.type.is_array:
                # Let other code handle array coercions
                pass
            else:
                node.value = nodes.CoercionNode(node.value, lhs_var.type)

        return node


class CudaCodeGenerator(_translate.LLVMCodeGenerator):
    def __init__(self, *args, **kws):
        super(CudaCodeGenerator, self).__init__(*args, **kws)
        self.__smem = {}

    def visit_CudaSMemArrayCallNode(self, node):
        from numba.ndarray_helpers import PyArrayAccessor
        ndarray_ptr_ty = node.variable.ltype
        ndarray_ty = ndarray_ptr_ty.pointee
        ndarray = self.builder.alloca(ndarray_ty)

        accessor = PyArrayAccessor(self.builder, ndarray)

        # store ndim
        store = lambda src, dst: self.builder.store(src, dst)
        accessor.ndim = Constant.int(llvm_types._int32, len(node.shape))


        # store data
        mod = self.builder.basic_block.function.module
        smem_elemtype = node.dtype.to_llvm(self.context)
        smem_type = Type.array(smem_elemtype, int(node.elemcount))
        smem = mod.add_global_variable(smem_type, 'smem', ADDRSPACE_SHARED)
        smem.initializer = Constant.undef(smem_type)

        smem_elem_ptr_ty = Type.pointer(smem_elemtype)
        smem_elem_ptr_ty_addrspace = Type.pointer(smem_elemtype,
                                                  ADDRSPACE_SHARED)
        smem_elem_ptr = smem.bitcast(smem_elem_ptr_ty_addrspace)
        tyname = str(smem_elemtype)
        tyname = {'float': 'f32', 'double': 'f64'}.get(tyname, tyname)
        s2g_intrinic = 'llvm.nvvm.ptr.shared.to.gen.p0%s.p3%s' % (tyname, tyname)
        shared_to_generic = mod.get_or_insert_function(
                                                       Type.function(smem_elem_ptr_ty,
                                                                     [smem_elem_ptr_ty_addrspace]),
                                                       s2g_intrinic)

        data = self.builder.call(shared_to_generic, [smem_elem_ptr])
        accessor.data = self.builder.bitcast(data, llvm_types._void_star)

        # store dims
        intp_t = llvm_types._intp
        const_intp = lambda x: Constant.int(intp_t, x)
        const_int = lambda x: Constant.int(Type.int(), x)

        dims = self.builder.alloca_array(intp_t,
                                         Constant.int(Type.int(),
                                                      len(node.shape)))

        for i, s in enumerate(node.shape):
            ptr = self.builder.gep(dims, map(const_int, [i]))
            store(const_intp(s), ptr)

        accessor.dims = dims

        # store strides
        strides = self.builder.alloca_array(intp_t,
                                            Constant.int(Type.int(),
                                                         len(node.strides)))

        for i, s in enumerate(node.strides):
            ptr = self.builder.gep(strides, map(const_int, [i]))
            store(const_intp(s), ptr)

        accessor.strides = strides

        return ndarray

    def visit_Name(self, node):
        try:
            return self.__smem[node.id]
        except KeyError:
            return super(CudaCodeGenerator, self).visit_Name(node)
    
    def visit_CudaSMemAssignNode(self, node):
        self.__smem[node.target.id] = value = self.visit(node.value)
        # preload
        var = node.target.variable
        acc = self.pyarray_accessor(value, var.type.dtype)
        if var.preload_data:
            var.preloaded_data = acc.data

        if var.preload_shape:
            shape = nodes.get_strides(self.builder, self.tbaa,
                                      acc.shape, var.type.ndim)
            var.preloaded_shape = tuple(shape)

        if var.preload_strides:
            strides = nodes.get_strides(self.builder, self.tbaa,
                                        acc.strides, var.type.ndim)
            var.preloaded_strides = tuple(strides)

#
# Helpers
#

def _make_sreg_call(attr):
    fname = sreg.SPECIAL_VALUES[attr]
    sig = minitypes.FunctionType(minitypes.uint32, [])
    funcnode = nodes.LLVMExternalFunctionNode(sig, fname)
    callnode = nodes.NativeFunctionCallNode(sig, funcnode, [])
    return callnode

def _make_sreg_pattern(x, y, z):
    x, y, z = (_make_sreg_call(i) for i in [x, y, z])
    mul = ast.BinOp(op=ast.Mult(), left=y, right=z)
    add = ast.BinOp(op=ast.Add(), left=x, right =mul)
    return add
