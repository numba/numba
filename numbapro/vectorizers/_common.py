# import tokenize, string
#
# from numba.vectorize._common import (
#     _llvm_ty_to_dtype,
#     GenericASTVectorize,
#     CommonVectorizeFromFunc,
#     post_vectorize_optimize,
# )
#
# #-------------------------------------------------------------------------------
# # gufunc signature parsing
#
# def parse_signature(sig):
#     '''Parse generalized ufunc signature.
#
#     NOTE: ',' (COMMA) is a delimiter; not separator.
#           This means trailing comma is legal.
#     '''
#     def stripws(s):
#         return ''.join(c for c in s if c not in string.whitespace)
#
#     def tokenizer(src):
#         return tokenize.generate_tokens(iter([src]).next)
#
#     def parse(src):
#         tokgen = tokenizer(src)
#         while True:
#             tok = tokgen.next()
#             if tok[1] == '(':
#                 symbols = []
#                 while True:
#                     tok = tokgen.next()
#                     if tok[1] == ')':
#                         break
#                     elif tok[0] == tokenize.NAME:
#                         symbols.append(tok[1])
#                     elif tok[1] == ',':
#                         continue
#                     else:
#                         raise ValueError('bad token in signature "%s"' % tok[1])
#                 yield tuple(symbols)
#                 tok = tokgen.next()
#                 if tok[1] == ',':
#                     continue
#                 elif tokenize.ISEOF(tok[0]):
#                     break
#             elif tokenize.ISEOF(tok[0]):
#                 break
#             else:
#                 raise ValueError('bad token in signature "%s"' % tok[1])
#
#     ins, outs = stripws(sig).split('->')
#     inputs = list(parse(ins))
#     outputs = list(parse(outs))
#
#     # check that all output symbols are defined in the inputs
#     isym = set()
#     osym = set()
#     for grp in inputs:
#         isym |= set(grp)
#     for grp in outputs:
#         osym |= set(grp)
#
#     diff = osym.difference(isym)
#     if diff:
#         raise NameError('undefined output symbols: %s' % ','.join(diff))
#
#     return inputs, outputs
