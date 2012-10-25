import sys
from numba.minivect import minitypes
from sreg import SPECIAL_VALUES
# modify numba behavior
from numba import utils, functions, ast_translate
from numba import visitors, nodes, error, ast_type_inference
from numba import pipeline
import ast as _ast

class CudaAttributeNode(nodes.Node):
    def __init__(self, value):
        self.value = value
    
    def resolve(self, name):
        return type(self)(getattr(self.value, name))
    
    def __repr__(self):
        return '<%s value=%s>' % (type(self).__nmae__, self.value)

class CudaSRegRewrite(visitors.NumbaTransformer,
                      ast_type_inference.NumpyMixin):
    
    def visit_Attribute(self, ast):
        from numbapro import cuda as _THIS_MODULE
        value = self.visit(ast.value)
        retval = ast # default to return the original node
        if isinstance(ast.value, _ast.Name):
            assert isinstance(value.ctx, _ast.Load)
            obj = self._myglobals.get(ast.value.id)
            if obj is _THIS_MODULE:
                retval = CudaAttributeNode(_THIS_MODULE).resolve(ast.attr)
            else:
                print ast.attr
                
        elif isinstance(value, CudaAttributeNode):
            retval = value.resolve(ast.attr)
        
        if retval.value in SPECIAL_VALUES:
            # replace with a MathCallNode
            sig = minitypes.FunctionType(minitypes.uint32, [])
            fname = SPECIAL_VALUES[retval.value]
            retval = nodes.MathCallNode(sig, [], None, name=fname)
        
        return retval

class NumbaproCudaPipeline(pipeline.Pipeline):
    def __init__(self, context, func, ast, func_signature, **kwargs):
        super(NumbaproCudaPipeline, self).__init__(context, func, ast,
                                                   func_signature, **kwargs)
        self.insert_specializer('rewrite_cuda_sreg', after='type_infer')
    
    
    def rewrite_cuda_sreg(self, ast):
        return CudaSRegRewrite(self.context, self.func, ast).visit(ast)


context = utils.get_minivect_context()  # creates a new NumbaContext
context.llvm_context = ast_translate.LLVMContextManager()
context.numba_pipeline = NumbaproCudaPipeline
function_cache = context.function_cache = functions.FunctionCache(context)
