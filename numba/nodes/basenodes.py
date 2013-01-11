from numba.nodes import *
import numba.nodes

class Node(ast.AST):
    """
    Superclass for Numba AST nodes
    """

    _fields = []
    _attributes = ('lineno', 'col_offset')

    def __init__(self, **kwargs):
        vars(self).update(kwargs)

    def _variable_get(self):
        if not hasattr(self, '_variable'):
            self._variable = Variable(self.type)

        return self._variable

    def _variable_set(self, variable):
        self._variable = variable

    variable = property(_variable_get, _variable_set)

    def coerce(self, dst_type):
        return numba.nodes.CoercionNode(self, dst_type)

    @property
    def cloneable(self):
        if isinstance(self, (CloneNode, CloneableNode)):
            return self
        return CloneableNode(self)


class Name(ast.Name, Node):

    cf_maybe_null = False
    raise_unbound_node = None

    _fields = ast.Name._fields + ('check_unbound',)

    def __init__(self, id, ctx, *args, **kwargs):
        super(Name, self).__init__(*args, **kwargs)
        self.id = self.name = id
        self.ctx = ctx

    def __repr__(self):
        type = getattr(self, 'type', "")
        if type:
            type = ', %s' % type
        name = self.name
        if hasattr(self, 'variable') and self.variable.renamed_name:
            name = self.variable.unmangled_name
        return "name(%s%s)" % (name, type)

    def __deepcopy__(self, memo):
        result = Name(self.id, self.ctx)
        result.cf_maybe_null = self.cf_maybe_null
        result.raise_unbound_node = self.raise_unbound_node
        return result

class WithPythonNode(Node):
    "with python: ..."

    _fields = ['body']

class WithNoPythonNode(WithPythonNode):
    "with nopython: ..."


class CloneableNode(Node):
    """
    Create a node that can be cloned. This allows sub-expressions to be
    re-used without re-evaluating them.
    """

    _fields = ['node']

    def __init__(self, node, **kwargs):
        super(CloneableNode, self).__init__(**kwargs)
        self.node = node
        self.clone_nodes = []
        self.type = node.type

    @property
    def clone(self):
        return CloneNode(self)

    def __repr__(self):
        return "cloneable(%s)" % self.node

class CloneNode(Node):
    """
    Clone a CloneableNode. This allows the node's sub-expressions to be
    re-used without re-evaluating them.

    The CloneableNode must be evaluated before the CloneNode is evaluated!
    """

    _fields = ['node']

    def __init__(self, node, **kwargs):
        super(CloneNode, self).__init__(**kwargs)

        assert isinstance(node, CloneableNode)
        self.node = node
        self.type = node.type
        node.clone_nodes.append(self)

        self.llvm_value = None

    @property
    def clone(self):
        self

    def __repr__(self):
        return "clone(%s)" % self.node

class ExpressionNode(Node):
    """
    Node that allows an expression to execute a bunch of statements first.
    """

    _fields = ['stmts', 'expr']

    def __init__(self, stmts, expr, **kwargs):
        super(ExpressionNode, self).__init__(**kwargs)
        self.stmts = stmts
        self.expr = expr
        self.type = expr.variable.type

    def __repr__(self):
        return "exprstat(..., %s)" % self.expr


class FunctionWrapperNode(Node):
    """
    This code is a wrapper function callable from Python using PyCFunction:

        PyObject *(*)(PyObject *self, PyObject *args)

    It unpacks the tuple to native types, calls the wrapped function, and
    coerces the return type back to an object.
    """

    _fields = ['body', 'return_result']

    def __init__(self, wrapped_function, signature, orig_py_func, fake_pyfunc,
                 orig_py_func_name):
        self.wrapped_function = wrapped_function
        self.signature = signature
        self.orig_py_func = orig_py_func
        self.fake_pyfunc = fake_pyfunc
        self.name = orig_py_func_name

