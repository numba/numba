import sys
from numba.minivect import minitypes
from sreg import SPECIAL_VALUES
# modify numba behavior
from numba import utils, functions, ast_translate
from numba import visitors, nodes, error, ast_type_inference
from numba import pipeline
import ast
from numba import ast_type_inference as type_inference
import logging
logger = logging.getLogger(__name__)

class CudaAttributeNode(nodes.Node):
    def __init__(self, value):
        self.value = value
    
    def resolve(self, name):
        return type(self)(getattr(self.value, name))
    
    def __repr__(self):
        return '<%s value=%s>' % (type(self).__nmae__, self.value)

class CudaSRegRewriteMixin(object):
    
    def visit_Attribute(self, node):
        from numbapro import cuda as _THIS_MODULE
        value = self.visit(node.value)
        retval = node # default to return the original node
        if isinstance(node.value, ast.Name):
            #assert isinstance(value.ctx, ast.Load)
            obj = self._myglobals.get(node.value.id)
            if obj is _THIS_MODULE:
                retval = CudaAttributeNode(_THIS_MODULE).resolve(node.attr)
            else:
                print node.attr
                
        elif isinstance(value, CudaAttributeNode):
            retval = value.resolve(node.attr)
        
        if retval.value in SPECIAL_VALUES:
            # replace with a MathCallNode
            sig = minitypes.FunctionType(minitypes.uint32, [])
            fname = SPECIAL_VALUES[retval.value]
            retval = nodes.MathCallNode(sig, [], None, name=fname)
            return retval
                
        if retval is node:
            retval = super(CudaSRegRewriteMixin, self).visit_Attribute(node)
        
        return retval

class CudaSharedRewriteMixin(object):
    pass

class CudaTypeInferer(CudaSRegRewriteMixin, CudaSharedRewriteMixin,
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
