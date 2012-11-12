'''
This module implements constant folding on the AST.  It handles simple cases
such as 
    
* 1 + 2 -> 3
* 2 ** 10 -> 1024
* N=1; N + 1 -> 2  (for N is assigned as global variable or a variable 
                    that's only assigned once)
'''
import operator, ast
from . import visitors

# shamelessly copied from Cython
compile_time_binary_operators = {
    '<'     : operator.lt,
    '<='    : operator.le,
    '=='    : operator.eq,
    '!='    : operator.ne,
    '>='    : operator.ge,
    '>'     : operator.gt,
    'is'    : operator.is_,
    'is_not': operator.is_not,
    '+'     : operator.add,
    '&'     : operator.and_,
    '/'     : operator.truediv,
    '//'    : operator.floordiv,
    '<<'    : operator.lshift,
    '%'     : operator.mod,
    '*'     : operator.mul,
    '|'     : operator.or_,
    '**'    : operator.pow,
    '>>'    : operator.rshift,
    '-'     : operator.sub,
    '^'     : operator.xor,
    'in'    : operator.contains,
    'not_in': lambda x, y: not operator.contains(x, y),
    'and'   : operator.and_,
    'or'    : operator.or_,
}

# shamelessly copied from Cython
compile_time_unary_operators = {
    'not'   : operator.not_,
    '~'     : operator.inv,
    '-'     : operator.neg,
    '+'     : operator.pos,
}

ast_to_binary_operator = {
    ast.Add     : '+',
    ast.Sub     : '-',
    ast.Mult    : '*',
    ast.Div     : '/',
    ast.FloorDiv: '//',
    ast.Pow     : '**',
    ast.LShift  : '<<',
    ast.RShift  : '>>',
    ast.BitOr   : '|',
    ast.BitAnd  : '&',
    ast.BitXor  : '^',
    ast.Lt      : '<',
    ast.LtE     : '<=',
    ast.Gt      : '>',
    ast.GtE     : '>=',
    ast.Eq      : '==',
    ast.NotEq   : '!=',
    ast.Is      : 'is',
    ast.IsNot   : 'is_not',
    ast.In      : 'in',
    ast.NotIn   : 'not_in',
    ast.And     : 'and',
    ast.Or      : 'or',
}

ast_to_unary_operator = {
    ast.Not     : 'not',
    ast.Invert  : '~',
    ast.USub    : '-',
    ast.UAdd    : '+',
}

class NotConstExprError(Exception):
    pass

class ConstantExprRecognizer(ast.NodeVisitor):
    def __init__(self, const_name_set):
        self.const_name_set = const_name_set

    def visit_BinOp(self, node):
        if not(self.visit(node.left) and self.visit(node.right)):
            raise NotConstExprError

    def visit_BoolOp(self, node):
        if not all(self.visit(x) for x in node.values):
            raise NotConstExprError

    def visit_Compare(self, node):
        if not(node.left and all(self.visit(x) for x in node.comparators)):
            raise NotConstExprError

    def generic_visit(self, node):
        if not is_constant(node, self.const_name_set):
            raise NotConstExprError

    def __call__(self, node):
        try:
            self.visit(node)
        except NotConstExprError as e:
            return False
        else:
            return True

class ConstantMarker(visitors.NumbaVisitor):
    '''A conservative constant marker.  Conservative because we handle the
    simplest cases only.
    '''
    def __init__(self, *args, **kws):
        super(ConstantMarker, self).__init__(*args, **kws)
        self._candidates = {}  # variable name -> value (rhs) node
        self._invalidated = set()

    def visit_Assign(self, node):
        targets = []
        for target in node.targets:
            targets.extend(self._flatten_aggregate(target))

        # targets contains tuple/list
        not_handled = len(node.targets) != len(targets)
                
        for target in targets:
            try:
                name = target.id
            except AttributeError:
                # Only for assignment into simple name on the LHS
                pass
            else:
                if name not in self._invalidated:
                    if not_handled or name in self._candidates:
                        self._invalidate(name)
                    else:
                        self._candidates[name] = node.value

    def visit_AugAssign(self, node):
        try:
            name = node.target.id
        except AttributeError:
            # Only for assignment into simple name on the LHS
            pass
        else:
            if name in self._candidates:
                self._invalidate(name)

    def visit_For(self, node):
        targets = self._flatten_aggregate(node.target)
        for t in targets:
            self._invalidate(t.id)
        for instr in node.body:
            self.visit(instr)

    def _flatten_aggregate(self, node):
        assert isinstance(node.ctx, ast.Store)
        if isinstance(node, ast.Tuple) or isinstance(node, ast.List):
            ret = []
            for i in node.elts:
                ret.extend(self._flatten_aggregate(i))
            return ret
        else:
            return [node]

    def _invalidate(self, name):
        self._invalidated.add(name)
        try:
            del self._candidates[name]
        except KeyError:
            pass

    def get_constants(self):
        '''Return a set of constant variable names
        '''
        const_names = set(self.varnames).difference(self._invalidated)
        const_names |= set(self.func_globals)

        constexpr_recognizer = ConstantExprRecognizer(const_names)
        retvals = []
        for k, v in self._candidates.items():
            if constexpr_recognizer(v):
                retvals.append(k)
        return set(retvals)

class ConstantFolder(visitors.NumbaTransformer):
    '''Perform constant folding on AST.

    NOTE: Forbids assignment to True, False, None.
    '''
    def __init__(self, *args, **kws):
        assert not hasattr(self, 'constvars') # not overwriting
        assert not hasattr(self, 'constvalues') # not overwriting
        self.constvars = kws.pop('constvars')
        self.constvalues = {}
        super(ConstantFolder, self).__init__(*args, **kws)

    def visit_BinOp(self, node):
        lval = node.left = self.visit(node.left)
        rval = node.right = self.visit(node.right)

        if self.is_constant(lval) and self.is_constant(rval):
            return self.eval_binary_operation(node.op, lval, rval)
        else:
            return node

    def visit_BoolOp(self, node):
        values = node.values = [self.visit(nd) for nd in node.values]
        if all(self.is_constant(v) for v in values):
            operation = lambda x, y: self.eval_binary_operation(node.op, x, y)
            return reduce(operation, values)
        else:
            return node

    def visit_Compare(self, node):
        left = node.left = self.visit(node.left)
        comparators = node.comparators = [self.visit(nd)
                                          for nd in node.comparators]
        operands = [left] + comparators
        operators = iter(reversed(node.ops))

        def operation(x, y):
            op = operators.next()
            return self.eval_binary_operation(op, x, y)

        if all(self.is_constant(nd) for nd in operands):
            return reduce(operation, operands)
        else:
            return node

    def visit_Assign(self, node):
        '''Store the rhs value so we can inline them in future reference.
            
        TODO: Remove assignment of constant
        '''
        # FIXME: does not handle assign to tuple
        names = []
        value = node.value = self.visit(node.value)
        for left in node.targets:
            try:
                name = left.id
            except AttributeError:
                return node # escape
            else:
                names.append(name)

        ct = 0
        for name in names:
            if name in self.constvars: # is known constant
                assert name not in self.constvalues
                self.constvalues[name] = value
                self.constvars.remove(name)
                ct += 1

        return node

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            try:
                return self.constvalues[node.id]
            except KeyError:
                pass
        return node

    def eval_binary_operation(self, op, left, right):
        '''Evaluate the constant expression and return a ast.Num instance
        containing the result.
        '''
        operator = ast_to_binary_operator[type(op)]
        func = compile_time_binary_operators[operator]
        ret = func(self.valueof(left), self.valueof(right))
        if ret is True:
            node = ast.Name(id='True', ctx=ast.Load())
        elif ret is False:
            node = ast.Name(id='False', ctx=ast.Load())
        elif ret is None:
            node = ast.Name(id='None', ctx=ast.Load())
        else:
            node = ast.Num(n=ret)
        return ast.copy_location(node, left)
    
    def valueof(self, node):
        '''Get constant value from AST node.
        '''
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id == 'True':
                return True
            elif node.id == 'False':
                return False
            elif node.id == 'None':
                return None
            elif node.id in self.constvalues:
                return self.valueof(self.constvalues[node.id])
            else:
                value = self.func_globals[node.id]
                if not is_simple_value(value):
                    raise ValueError("%s is not a simple value.")
                return value
        raise ValueError("node %s is not a has constant value" % node)

    def is_constant(self, node):
        globals = set(self.func_globals).difference(self.local_names)
        return is_constant(node, globals | set(self.constvalues))

def is_constant(node, constants=set()):
    if isinstance(node, ast.Num):
        return True
    elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
        if node.id in ['True', 'False', 'None']:
            return True
        elif node.id in constants:
            return True
    return False

def is_simple_value(value):
    return (   isinstance(value, int)
            or isinstance(value, long)
            or isinstance(value, float)
            or value is True
            or value is False
            or value is None)
