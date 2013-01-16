"""
This module provides deferred types used for type dependences that
haven't been processed yet, or circular dependences.

1) Types participating in statements that are deferred until later:

        x = 0                   # x_0
        for i in range(10):
            # x_1 = phi(x_0, x_2)
            print x
            x = i               # x_2

    Here x_2 is not resolved yet when we encounter the print statement.

2) Types participating in type graph cycles:

        x = 0                   # x_0
        for i in range(10):
            # x_1 = phi(x_0, x_2)
            print x
            x = x + i           # x_2

    Here we have:

        type(x_1) = promote(type(x_0) = int, type(x_2))
        type(x_2) = promote(type(x_1), type(i) = Py_ssize_t)

    This is a simple cycle that will constitute the following graph:

        x_0 (int)
            \
              x_1 __
                \   \
                  X_2
                    \
                      i_0 (Py_ssize_t)
"""

from numba import oset
from numba.typesystem import *

class UninitializedType(NumbaType):

    is_uninitialized = True
    subtypes = ['base_type']

    def __init__(self, base_type, **kwds):
        super(UninitializedType, self).__init__(**kwds)
        self.base_type = base_type

    def to_llvm(self, context):
        ltype = self.base_type.to_llvm(context)
        return ltype

    def __repr__(self):
        return "<uninitialized>"

class UnresolvedType(NumbaType):
    """
    The directed type graph works as follows:

        1) if type x depends on type y, then y is a parent of x.
        2) we construct a condensation graph by contracting strongly connected
           components to single nodes
        3) we resolve types in topological order
               -> types in SCCs are handled specially
    """

    is_unresolved = True
    rank = 1

    def __init__(self, variable, **kwds):
        super(UnresolvedType, self).__init__(**kwds)
        self.variable = variable
        self.assertions = []
        self.parents = oset.OrderedSet()
        self.children = oset.OrderedSet()

    def add_children(self, children):
        for child in children:
            if child.is_unresolved:
                self.children.add(child)
                child.parents.add(self)

    def add_parents(self, parents):
        for parent in parents:
            if parent.is_unresolved:
                self.parents.add(parent)
                parent.children.add(self)

    def __hash__(self):
        return hash(self.variable)

    def __eq__(self, other):
        return (isinstance(other, UnresolvedType) and
                self.variable == other.variable and
                self.is_deferred == other.is_deferred and
                self.is_promotion == other.is_promotion and
                self.is_unanalyzable == other.is_unanalyzable)

    def simplify(self):
        return not (self.resolve() is self)

    def make_assertion(self, assertion_attr, node, msg):
        def assertion(result_type):
            if not getattr(result_type, assertion_attr):
                raise error.NumbaError(node, msg)
        self.assertions.append(assertion)

    def process_assertions(self, result_type):
        for assertion in self.assertions:
            assertion(result_type)

        del self.assertions[:]

    def resolve(self):
        if not self.variable.type:
            self.variable.type = self
        result = self.variable.type
        if not result.is_unresolved:
            self.process_assertions(result)
        return result

class PromotionType(UnresolvedType):

    is_promotion = True
    resolved_type = None

    count = 0 # for debugging

    def __init__(self, variable, context, types, assignment=False, **kwds):
        super(PromotionType, self).__init__(variable, **kwds)
        self.context = context
        self.types = types
        self.assignment = assignment
        variable.type = self

        self.add_parents(type for type in types if type.is_unresolved)

        self.count = PromotionType.count
        PromotionType.count += 1

    @property
    def t(self):
        # for debugging only
        return list(self.types)

    def add_type(self, seen, type, types):
        if type not in seen:
            if type.is_unresolved:
                seen.add(type)
                new_type = type.resolve()
                if new_type is not type:
                    seen.add(new_type)
                    self.add_type(seen, new_type, types)
                    type = new_type
                else:
                    types.add(type)
            else:
                types.add(type)

            return type

    def dfs(self, types, seen):
        for type in self.types:
            if type not in seen:
                seen.add(type)
                type = resolve_type_chain(type)
                seen.add(type)
                if type.is_promotion:
                    type.dfs(types, seen)
                elif not type.is_uninitialized:
                    types.add(type)

    def find_types(self, seen):
        types = set([self])
        seen.add(self)
        seen.add(self.variable.deferred_type)
        self.dfs(types, seen)
        types.remove(self)
        return types

    def find_simple(self, seen):
        types = set()
        for type in self.types:
            if type.is_promotion:
                types.add(type.types)
            else:
                type.add(type)

        return types

    def get_partial_types(self, unresolved_types):
        for unresolved_type in unresolved_types:
            if (unresolved_type.is_reanalyse_circular and
                    unresolved_type.resolved_type):
                unresolved_types.append(unresolved_type)

    def _simplify(self, seen=None):
        """
        Simplify a promotion type tree:

            promote(int_, float_)
                -> float_

            promote(deferred(x), promote(float_, double), int_, promote(<self>))
                -> promote(deferred(x), double)

            promote(deferred(x), deferred(y))
                -> promote(deferred(x), deferred(y))
        """
        if seen is None:
            seen = set()

        # Find all types in the type graph and eliminate nested promotion types
        types = self.find_types(seen)
        # types = self.find_simple(seen)

        resolved_types = [type for type in types if not type.is_unresolved]
        unresolved_types = [type for type in types if type.is_unresolved]
        self.get_partial_types(unresolved_types)

        self.variable.type = self
        if not resolved_types:
            # Everything is deferred
            self.resolved_type = None
            return False
        else:
            # Simplify as much as possible
            if self.assignment:
                result_type, unresolved_types = promote_for_assignment(
                        self.context, resolved_types, unresolved_types,
                        self.variable.name)
            else:
                result_type = promote_for_arithmetic(self.context, resolved_types)

            self.resolved_type = result_type
            if len(resolved_types) == len(types) or not unresolved_types:
                self.variable.type = result_type
                return True
            else:
                old_types = self.types
                self.types = set([result_type] + unresolved_types)
                return old_types != self.types

    def simplify(self, seen=None):
        try:
            return self._simplify(seen)
        except minierror.UnpromotableTypeError, e:
            if self.variable.name:
                name = "variable %s" % self.variable.name
            else:
                name = "subexpression"

            types = sorted(e.args[0], key=str)
            types = tuple(types)
            raise TypeError("Cannot promote types %s for %s" % (types, name))

    @classmethod
    def promote(cls, *types):
        var = Variable(None)
        type = PromotionType(var, types)
        type.resolve()
        return type.variable.type

    repr_seen = None
    repr_count = 0

    def __repr__(self):
        if not self.repr_seen:
            self.repr_seen = set()

        self.repr_seen.add(self)
        self.repr_count += 1

        types = []
        for type in self.types:
            if type not in self.repr_seen:
                types.append(type)
                self.repr_seen.add(type)
            else:
                types.append("...")

        result = "promote%d(%s)" % (self.count, ", ".join(map(str, types)))
        self.repr_count -= 1
        if not self.repr_count:
            self.repr_seen = None

        return result

class DeferredType(UnresolvedType):
    """
    We don't know what the type is at the point we need a type, so we create
    a deferred type.

        Depends on: self.variable.type

    Example:

        def func():
            for i in range(10):
                # type(x) = phi(undef, deferred(x_1)) = phi(deferred(x_1))
                if i > 1:
                    print x   # type is deferred(x_1)
                x = ...       # resolve deferred(x_1) to type(...)
    """

    is_deferred = True
    updated = False

    def update(self):
        assert self.variable.type is not self
        self.updated = True

        type = self.variable.type
        if not type.is_unresolved:
            # Type is a scalar or otherwise resolved type tree, and doesn't
            # need to participate in the graph
            return

        for parent in self.parents:
            if self in parent.children:
                parent.children.remove(self)
            parent.children.add(type)

        for child in self.children:
            if self in child.parents:
                child.parents.remove(self)
            child.parents.add(type)

        type.parents.update(self.parents)
        type.children.update(self.children)

#    def resolve(self):
#        result_type = super(DeferredType, self).resolve()
#        if result_type is not self and result_type.is_unresolved:
#            result_type = result_type.resolve()
#        self.variable.type = result_type
#        return result_type

    def __repr__(self):
        if self.variable.type is self:
            return "<deferred(%s)>" % (self.variable.unmangled_name,)

        return "<deferred(%s)>" % self.variable.type

#    def to_llvm(self, context):
#        assert self.resolved_type, self
#        return self.resolved_type.to_llvm(context)

class ReanalyzeCircularType(UnresolvedType):
    """
    This is useful when there is a circular dependence on yourself. e.g.

        s = "hello"
        for i in range(5):
            s = s[1:]

    The type of 's' depends on the result of the slice, and on the input to
    the loop. But to determine the output, we need to assume the input,
    and unify the output with the input, and see the result for a subsequent
    slice. e.g.

        a = np.empty((10, 10, 10))
        for i in range(3):
            a = a[0]

    Here the type would change on each iteration. Arrays do not demote to
    object, but other types do. The same goes for a call:

        for i in range(n):
            f = f(i)

    but also

        x = 0
        for i in range(n):
            x = f(x)

    or linked-list traversal

        current = ...
        while current:
            current = current.next
    """

    is_reanalyze_circular = True
    resolved_type = None
    converged = False

    def __init__(self, variable, type_inferer, **kwds):
        super(ReanalyzeCircularType, self).__init__(variable, **kwds)
        self.type_inferer = type_inferer
        self.dependences = []

    def update(self):
        "Update the graph after having updated the dependences"
        self.add_parents(node.variable.type
                            for node in self.dependences
                                if node.variable.type.is_unresolved)

    def _reinfer(self):
        result_type = self.retry_infer()
        if not result_type.is_unresolved:
            self.resolved_type = result_type
            self.variable.type = result_type

        return result_type is not self

    def substitute_and_reinfer(self):
        """
        Try substituting resolved parts of promotions and reinfer the types.
        """
        from numba import symtab

        if not self.variable.type.is_unresolved:
            return False

        # Find substitutions and save original variables
        old_vars = []
        for node in self.dependences:
            sub_type = self.substitution_candidate(node.variable)
            if sub_type:
                old_vars.append((node, node.variable))
                node.variable = symtab.Variable(sub_type, name='<substitute>')

        if old_vars:
            # We have some substitutions, retry type inference
            result = self._reinfer()
            # Reset our original variables!
            for node, old_var in old_vars:
                node.variable = old_var
            return result

        # We cannot substitute any promotion candidates, see if we can resolve
        # anyhow (this should be a cheap operation anyway if it fails)
        new_type = self.retry_infer()
        if not new_type.is_unresolved:
            self.variable.type = new_type
            return True

        return False

    def substitution_candidate(self, variable):
        if variable.type.is_unresolved:
            variable.type.resolve()

        if variable.type.is_promotion:
            p = resolve_var(variable)
            if p.is_promotion and p.resolved_type:
                return p.resolved_type

        return None

    def simplify(self):
        """
        Resolve the reanalyzable statement by setting the already resolved
        dependences for the type inference code.
        """
        if self.resolved_type is not None:
            return False # nothing changed

        for dep in self.dependences:
            if dep.variable.type.is_unresolved:
                dep.variable.type = dep.variable.type.resolve()
            assert not dep.variable.type.is_unresolved

        return self._reinfer()

    def retry_infer(self):
        "Retry inferring the type with the new type set"

    def substitute_variables(self, substitutions):
        "Try to set the new variables and retry type inference"


class DeferredIndexType(ReanalyzeCircularType):
    """
    Used when we don't know the type of the variable being indexed.
    """

    def __init__(self, variable, type_inferer, index_node, **kwds):
        super(DeferredIndexType, self).__init__(variable, type_inferer, **kwds)
        self.type_inferer = type_inferer
        self.index_node = index_node

    def retry_infer(self):
        node = self.type_inferer.visit_Subscript(self.index_node,
                                                 visitchildren=False)
        return node.variable.type

    def __repr__(self):
        return "<deferred_index(%s, %s)" % (self.index_node,
                                            ", ".join(map(str, self.parents)))

class DeferredCallType(ReanalyzeCircularType):
    """
    Used when we don't know the type of the expression being called, or when
    we have an autojitting function and don't know all the argument types.
    """

    def __init__(self, variable, type_inferer, call_node, **kwds):
        super(DeferredCallType, self).__init__(variable, type_inferer, **kwds)
        self.type_inferer = type_inferer
        self.call_node = call_node

    def retry_infer(self):
        node = self.type_inferer.visit_Call(self.call_node,
                                            visitchildren=False)
        return node.variable.type

    def __repr__(self):
        return "<deferred_call(%s, %s)" % (self.call_node,
                                           ", ".join(map(str, self.parents)))

def resolve_type_chain(type):
    if not type.is_unresolved:
        return type

    while type.is_unresolved:
        old_type = type
        type = old_type.resolve()
        if type is old_type or not type.is_unresolved:
            break

    return type

def error_circular(var):
    raise error.NumbaError(
        var.name_assignment and var.name_assignment.assignment_node,
        "Unable to infer type for assignment to %r,"
        " insert a cast or initialize the variable." % var.name)

class StronglyConnectedCircularType(UnresolvedType):
    """
    Circular type dependence. This can be a strongly connected component
    of just promotions, or a mixture of promotions and re-inferable statements.

    If we have only re-inferable statements, but no promotions, we have nothing
    to feed into the re-inference process, so we issue an error.
    """

    is_resolved = False
    is_scc = True

    def __init__(self, scc, **kwds):
        super(StronglyConnectedCircularType, self).__init__(None, **kwds)
        self.scc = scc

        types = set(scc)
        for type in scc:
            self.add_children(type.children - types)
            self.add_parents(type.parents - types)

        self.types = scc
        self.promotions = set(type for type in scc if type.is_promotion)
        self.reanalyzeable = set(type for type in scc if type.is_reanalyze_circular)

    def retry_infer_reanalyzable(self):
        for reanalyzeable in self.reanalyzeable:
            if reanalyzeable.resolve().is_unresolved:
                reanalyzeable.substitute_and_reinfer()

    def err_no_input(self):
        raise error.NumbaError(self.variable and self.variable.assignment_node,
                               "No input types for this assignment were "
                               "found, a cast is needed")

    def retry_infer(self):
        candidates = []
        no_input = []
        for promotion in self.promotions:
            p = resolve_var(promotion.variable)
            if p.is_promotion:
                if p.resolved_type:
                    candidates.append(p)
                else:
                    no_input.append(p)

        if not candidates:
            if no_input:
                self.err_no_input()

            # All types are resolved, resolve all delayed types
            self.retry_infer_reanalyzable()
            return

        # Re-infer re-analyzable statements until we converge
        changed = True
        while changed:
            self.retry_infer_reanalyzable()
            changed = False
            for p in list(self.promotions):
                if p.resolve() is not p:
                    self.promotions.remove(p)
                else:
                    changed |= p.simplify()

        for promotion in self.promotions:
            promotion.variable.type = promotion.resolved_type

    def resolve_promotion_cycles(self):
        p = self.promotions.pop()
        self.promotions.add(p)

        p.simplify()
        result_type = p.resolve()

        if result_type.is_unresolved:
            # Note: There are no concrete input types and it is impossible to
            #       infer anything but 'object'. Usually this indicates an
            #       invalid program
            error_circular(result_type.variable)

        for p in self.promotions:
            p.variable.type = result_type

    def simplify(self):
        if self.reanalyzeable:
            self.retry_infer()
        elif self.promotions:
            self.resolve_promotion_cycles()
        else:
            assert False

        self.is_resolved = True

    def resolve(self):
        # We don't have a type, we are only an aggregation of circular types
        raise TypeError

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        return id(self) == id(other)


def dfs(start_type, stack, seen, graph=None, parents=False):
    seen.add(start_type)
    if parents:
        children = start_type.parents
    else:
        children = start_type.children

    for child_type in children:
        if child_type not in seen and child_type.is_unresolved:
            if graph is None or child_type in graph:
                dfs(child_type, stack, seen, graph, parents=parents)

    stack.append(start_type)


class UnanalyzableType(UnresolvedType):
    """
    A type that indicates the statement cannot be analyzed without first
    analysing its dependencies.
    """

    is_unanalyzable = True

def resolve_var(var):
    if var.type.is_unresolved:
        var.type.simplify()
    if var.type.is_unresolved:
        var.type = var.type.resolve()

    return var.type


def kosaraju_strongly_connected(start_type, strongly_connected, seen):
    """
    Find the strongly connected components in the connected graph starting at
    start_type.
    """
    stack = []
    dfs(start_type, stack, set(seen))

    seen = set(seen)
    graph = oset.OrderedSet(stack)
    while stack:
        start = stack[-1]
        scc = []
        dfs(start, scc, seen, graph, parents=True)
        if len(scc) > 1:
            scc_type = StronglyConnectedCircularType(scc)
            for type in scc_type.types:
                strongly_connected[type] = scc_type
                stack.pop()
        else:
            strongly_connected[scc[0]] = scc[0]
            stack.pop()

#------------------------------------------------------------------------
# Type promotion and validation
#------------------------------------------------------------------------

def _validate_array_types(array_types):
    first_array_type = array_types[0]
    for array_type in array_types[1:]:
        if array_type.ndim != first_array_type.ndim:
            raise TypeError(
                "Cannot unify arrays with distinct dimensionality: "
                "%d and %d" % (first_array_type.ndim, array_type.ndim))
        elif array_type.dtype != first_array_type.dtype:
            raise TypeError("Cannot unify arrays with distinct dtypes: "
                            "%s and %s" % (first_array_type.dtype,
                                           array_type.dtype))

def promote_for_arithmetic(context, types, assignment=False):
    result_type = types[0]
    for type in types[1:]:
        result_type = context.promote_types(result_type, type, assignment)

    return result_type

def promote_arrays(array_types, non_array_types, types,
                   unresolved_types, var_name):
    """
    This promotes arrays for assignments. Arrays must have a single consistent
    type in an assignment (phi). Any promotion of delayed types are immediately
    resolved.
    """
    _validate_array_types(array_types)

    # TODO: figure out whether result is C/F/inner contig
    result_type = array_types[0].strided

    def assert_equal(other_type):
        if result_type != other_type:
            raise TypeError(
                "Arrays must have consistent types in assignment "
                "for variable %r: '%s' and '%s'" % (
                    var_name, result_type, other_type))

    if len(array_types) < len(types):
        assert_equal(non_array_types[0])

    # Add delayed assertion that triggers when the delayed types are resolved
    for unresolved_type in unresolved_types:
        unresolved_type.assertions.append(assert_equal)

    return result_type, []

def promote_for_assignment(context, types, unresolved_types, var_name):
    """
    Promote a list of types for assignment (e.g. in a phi node).

        - if there are any objects, the result will always be an object
        - if there is an array, all types must be of that array type
              (minus any contiguity constraints)
    """
    obj_types = [type for type in types if type == object_ or type.is_array]
    if obj_types:
        array_types = [obj_type for obj_type in obj_types if obj_type.is_array]
        non_array_types = [type for type in types if not type.is_array]
        if array_types:
            return promote_arrays(array_types, non_array_types, types,
                                  unresolved_types, var_name)
        else:
            # resolved_types = obj_types
            return object_, []

    partial_result_type = promote_for_arithmetic(context, types,
                                                 assignment=True)
    return partial_result_type, unresolved_types
