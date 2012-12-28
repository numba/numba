from numba import decorators, pipeline, ast_translate

from numbapro import array_expressions, array_slicing, prange

class NumbaproPipeline(pipeline.Pipeline):
    """
    Pipeline that supports NumbaPro specific functionality
    """

    def __init__(self, context, func, ast, func_signature, **kwargs):
        super(NumbaproPipeline, self).__init__(context, func, ast,
                                               func_signature, **kwargs)
        self.try_insert_specializer('rewrite_array_expressions',
                                    before='specialize')

        self.insert_specializer('expand_prange',
                                before='cfg')
        self.insert_specializer('rewrite_prange_privates',
                                before='cfg')
        self.insert_specializer('fix_ast_locations',
                                before='cfg')
        self.insert_specializer('cleanup_prange',
                                after='type_infer')

    def rewrite_array_expressions(self, ast):
        # transformer = ArrayExpressionRewriteUfunc(self.context, self.func, ast)
        transformer = self.make_specializer(
            array_expressions.ArrayExpressionRewriteNative, ast)
        return transformer.visit(ast)

    def expand_prange(self, ast):
        transformer = self.make_specializer(prange.PrangeExpander, ast)
        return transformer.visit(ast)

    def rewrite_prange_privates(self, ast):
        transformer = self.make_specializer(prange.PrangePrivatesReplacer, ast)
        return transformer.visit(ast)

    def cleanup_prange(self, ast):
        transformer = self.make_specializer(prange.PrangeCleanup, ast)
        return transformer.visit(ast)


class NumbaProCodegen(array_slicing.NativeSliceCodegenMixin):
    """
    Support native slicing code generation and prange.
    """

NumbaproPipeline.add_mixin('codegen', NumbaProCodegen, before=True)
decorators.context.numba_pipeline = NumbaproPipeline
