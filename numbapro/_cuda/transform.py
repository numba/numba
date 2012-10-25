import sys
from numba.minivect import minitypes
from .sreg import SPECIAL_VALUES
from . import smem
# modify numba behavior
from numba import utils, functions, ast_translate
from numba import visitors, nodes, error, ast_type_inference
from numba import pipeline
import ast
from numba import ast_type_inference as type_inference
import logging
logger = logging.getLogger(__name__)

class CudaAttributeNode(nodes.Node):
    _attributes = ['value']
    
    def __init__(self, value):
        self.value = value
    
    def resolve(self, name):
        return type(self)(getattr(self.value, name))
    
    def __repr__(self):
        return '<%s value=%s>' % (type(self).__nmae__, self.value)

class CudaSMemArrayNode(nodes.Node):
    pass

class CudaSMemArrayCallNode(nodes.Node):
    def __init__(self, shape, dtype):
        self.shape = shape
        self.type = minitypes.ArrayType(dtype=dtype,
                                        ndim=len(self.shape),
                                        is_c_contig=True)

class CudaAttrRewriteMixin(object):
    
    def visit_Attribute(self, node):
        from numbapro import cuda as _THIS_MODULE
        value = self.visit(node.value)
        retval = node # default to return the original node
        if isinstance(node.value, ast.Name):
            #assert isinstance(value.ctx, ast.Load)
            obj = self._myglobals.get(node.value.id)
            if obj is _THIS_MODULE:
                retval = CudaAttributeNode(_THIS_MODULE).resolve(node.attr)
        elif isinstance(value, CudaAttributeNode):
            retval = value.resolve(node.attr)
        
        if retval.value in SPECIAL_VALUES:  # sreg
            # replace with a MathCallNode
            sig = minitypes.FunctionType(minitypes.uint32, [])
            fname = SPECIAL_VALUES[retval.value]
            retval = nodes.MathCallNode(sig, [], None, name=fname)
        elif retval.value == smem._array:   # allocate shared memory
            retval = CudaSMemArrayNode()

        if retval is node:
            retval = super(CudaAttrRewriteMixin, self).visit_Attribute(node)
        
        return retval

    def visit_Call(self, node):
        func = self.visit(node.func)
        if isinstance(func, CudaSMemArrayNode):
            kws = dict((kw.arg, kw.value)for kw in node.keywords)

            shape = tuple()
            for elem in kws['shape'].elts:
                shape += (elem.n,)  # FIXME: must be a constant
            
            dtype_id = kws['dtype'].id # FIXME must be a ast.Name
            dtype = self._myglobals[dtype_id] # must be a Numba type
        
            return CudaSMemArrayCallNode(shape=shape, dtype=dtype)
        else:
            return super(CudaAttrRewriteMixin, self).visit_Call(node)

    def visit_Assign(self, node):
        rhs = self.visit(node.value)
        if isinstance(rhs, CudaSMemArrayCallNode):
            raise
        else:
            return super(CudaAttrRewriteMixin, self).visit_Assign(node)
                      
class CudaTypeInferer(CudaAttrRewriteMixin, 
                      type_inference.TypeInferer):
    pass

class NumbaproCudaPipeline(pipeline.Pipeline):
#    def __init__(self, context, func, node, func_signature, **kwargs):
#        super(NumbaproCudaPipeline, self).__init__(context, func, node,
#                                                   func_signature, **kwargs)
#        self.insert_specializer('rewrite_cuda_sreg', after='type_infer')
#    
    def rewrite_cuda_sreg(self, node):
        return CudaSRegRewrite(self.context, self.func, node).visit(node)

    def type_infer(self, node):
        type_inferer = self.make_specializer(CudaTypeInferer, node,
                                             locals=self.locals)
        type_inferer.infer_types()
        self.func_signature = type_inferer.func_signature
        logger.debug("signature for %s: %s" % (self.func.func_name,
                                               self.func_signature))
        self.symtab = type_inferer.symtab
        return node

context = utils.get_minivect_context()  # creates a new NumbaContext
context.llvm_context = ast_translate.LLVMContextManager()
context.numba_pipeline = NumbaproCudaPipeline
function_cache = context.function_cache = functions.FunctionCache(context)
