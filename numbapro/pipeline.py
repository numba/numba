from numba import decorators, pipeline, ast_translate

from numbapro import array_expressions, array_slicing, prange

class NumbaproPipeline(pipeline.Pipeline):
    """
    Pipeline that support
    """

    def __init__(self, context, func, ast, func_signature, **kwargs):
        super(NumbaproPipeline, self).__init__(context, func, ast,
                                               func_signature, **kwargs)
        self.try_insert_specializer('rewrite_array_expressions',
                                    after='specialize')
#        self.try_insert_specializer('rewrite_prange_privates',
#                                    after='closure_type_inference')

    def rewrite_array_expressions(self, ast):
        # transformer = ArrayExpressionRewriteUfunc(self.context, self.func, ast)
        transformer = self.make_specializer(
            array_expressions.ArrayExpressionRewriteNative, ast,
            llvm_module=self.llvm_module)
        return transformer.visit(ast)

    def rewrite_prange_privates(self, ast):
        transformer = self.make_specializer(prange.PrangePrivatesReplacer, ast)
        return transformer.visit(ast)


class NumbaProTypeInferer(prange.PrangeTypeInfererMixin):
    """
    NumbaPro enhancements to numba type inference.
    """

class NumbaProCodegen(array_slicing.NativeSliceCodegenMixin,
                      prange.PrangeCodegenMixin):
    """
    Support native slicing code generation and prange.
    """

NumbaproPipeline.add_mixin('type_infer', NumbaProTypeInferer, before=True)
NumbaproPipeline.add_mixin('codegen', NumbaProCodegen)
decorators.context.numba_pipeline = NumbaproPipeline