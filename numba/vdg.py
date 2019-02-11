"""
Define an value-dependency graph analysis.


"""
from collections import defaultdict, Iterable
from pprint import pprint
import textwrap

from numba import ir


class VarDependencyGraph(object):
    """Var-dependency graph

    An enhanced def-use chains.

    This is a sparse representation of the IR.  Only variable definitions and
    use-points of variables are recorded.  CFG information is mostly lost
    (except for loops that are revealed as variables with self-dependency).

    Attributes
    ----------
    - _defpoints: var -> set[assignment]
        Map for var definitions.
    - _usepoints: var -> set[inst]
        Map for var use points.
    - _depmap: var -> set[var]
        Map for var dependency.
        The values are vars that the associated key may depends on.
    """
    @staticmethod
    def from_blocks(blocks):
        """Given the blocks, build a use-def tree.
        """
        iter_inst = (inst for blk in blocks.values() for inst in blk.body)

        defpoints = defaultdict(set)
        usepoints = defaultdict(set)
        depmap = {}

        # Build use-def
        for inst in iter_inst:
            if isinstance(inst, ir.Assign):
                target = inst.target
                defpoints[target].add(inst)
            for v in inst.used:
                usepoints[v].add(inst)

        # Build dependency
        for var, assignset in defpoints.items():
            depends = {x.target
                       for assign in assignset
                       for use in assign.used
                       for x in defpoints[use]}
            depmap[var] = depends

        vdg = VarDependencyGraph(
            defpoints=defpoints,
            usepoints=usepoints,
            depmap=depmap,
        )
        return vdg

    def __init__(self, defpoints, usepoints, depmap):
        self._defpoints = defpoints
        self._usepoints = usepoints
        self._depmap = depmap
        self._scope = next(iter(defpoints.keys())).scope
        # caches
        self._cache_preds = {}
        self._cache_descs = {}

    def _iterative_search(self, var, cache, functor):
        """Search the graph until the the result set stop growing.

        Parameters
        ----------
        var : str or numba.ir.Var
        cache : dict
            the cache
        fucntor : callable with signature (var) -> set[var]
            A callable to determine the neighbours to look at given
            the current var.  The output from it is joined with the result
            set.
        """
        var = self._norm_var(var)
        # Check cache early
        if var in cache:
            return cache[var]
        # Walk the graph depth-first
        pending = [var]
        seen = set()
        res = set()
        while pending:
            cur = pending.pop()
            seen.add(cur)
            if cur in cache:
                new = cache[cur]
            else:
                new = functor(cur)
                pending.extend(new - seen)
            res |= new
        return res

    def get_uses(self, var):
        """Returns a set of the use points of the *var*.
        """
        var = self._norm_var(var)
        return self._usepoints[var]

    def get_definitions(self, var):
        """Returns a set of the defining points of the *var*.
        """
        var = self._norm_var(var)
        return self._defpoints[var]

    def get_uses_no_alias(self, var):
        """Returns a set of the use points of the *var*.
        """
        uses = self.get_uses(var)
        progress = True
        while progress:
            new = set()
            progress = False
            for u in uses:
                if isinstance(u.value, ir.Var):
                    new |= self.get_uses(u.target)
                else:
                    new.add(u)
            defs = new
        return defs

    def get_defs_no_alias(self, var):
        defs = self.get_definitions(var)
        progress = True
        while progress:
            new = set()
            progress = False
            for d in defs:
                if isinstance(d.value, ir.Var):
                    new |= self.get_definitions(d.value)
                    progress = True
                else:
                    new.add(d)
            defs = new
        return defs

    def get_predecessors(self, var):
        """Returns a set of predecessors of `var` in the dependency graph.
        `var` *maybe* a dependent of the returned variables.

        Parameters
        ----------
        var : str or numba.ir.Var
        """
        return self._iterative_search(
            var,
            self._cache_preds,
            lambda x: self._depmap[x],
        )

    def get_descendants(self, var):
        """Returns a set of descendants of `var` in the dependency graph.
        `var` *maybe* a dependency of the returned variables.

        Parameters
        ----------
        var : str or numba.ir.Var
        """
        return self._iterative_search(
            var,
            self._cache_descs,
            lambda x: {s.target for s in self._usepoints[x]
                       if isinstance(s, ir.Assign)},
        )

    def get_roots(self):
        """Returns a set of variables that are the root of the dependency
        graph.  These variables do not depend on any other variables.
        They are usually the arguments and globals.
        """
        roots = set()
        excluded = set()
        for k in self._defpoints:
            if k not in excluded:
                if not self.get_predecessors(k):
                    roots.add(k)
                excluded |= self.get_descendants(k)
        return roots

    def find_self_dependents(self):
        """Returns a set of  vars that are self-dependent.

        These are variables modified in loops.
        """
        # Walk the graph from the roots and keep track of vars with
        # dependency that have been visited.
        roots = list(self.get_roots())

        loops = set()

        def walker(cur, path):
            children = self.get_descendants(cur)
            for c in children:
                if c in path:
                    yield c
                else:
                    yield walker(c, path + [c])

        # Start the walkers
        tmps = [walker(cur, [cur]) for cur in roots]

        # Reduce the walkers
        while tmps:
            out = []
            for it in tmps:
                if isinstance(it, Iterable):
                    out.extend(it)
                else:
                    loops.add(it)
            tmps = out
        return loops

    def _norm_var(self, var):
        """Normalize a `str` or `numba.ir.Var` into a `numba.ir.Var`
        """
        if isinstance(var, ir.Var):
            return var
        else:
            return self._scope.get(var)

    def show_graph(self, name='vdg', view=True):
        """Create a Graphviz DOT representation.

        - Vars are in bold rectangle boxes
        - Operations are in rounded boxes
        - Usage only operations are in grayed rounded boxes.

        """
        from graphviz import Digraph

        g = Digraph(name=name)
        # Helper to compute a identifier for a node
        nodeid = lambda x: hex(id(x))
        # Buffer to delay drawing statements
        delay_buf = set()

        # Follow the dependency
        for k, vs in self._depmap.items():
            # Draw nodes
            g.node(nodeid(k), label=str(k), shape='rect', penwidth='2')
            for v in vs:
                g.node(nodeid(v), label=str(v), shape='rect', penwidth='2')
            # Draw edges showing the assignment of the predecessors
            for v in vs:
                # For every producer
                for assign in self._defpoints[k]:
                    # Does the producer uses `v`?
                    if v in assign.used:
                        delay_buf.add(assign)
                        g.edge(nodeid(v), nodeid(assign))

        # Draw and connect all definitions to there defining variables
        for k, vs in self._defpoints.items():
            for assign in vs:
                delay_buf.add(assign)
                g.edge(nodeid(assign), nodeid(k))

        def draw_stmt(g, stmt, **kwargs):
            lines = textwrap.wrap(
                str(stmt), width=80, subsequent_indent='  ',
            )
            lines = [ln + '\l' for ln in lines] + ['({})'.format(stmt.loc)]
            wrapped = ''.join(lines)
            g.node(nodeid(stmt), label=wrapped, **kwargs)

        # Draw delayed
        for stmt in delay_buf:
            kwargs = {}
            if isinstance(getattr(stmt, 'value', None), (ir.Global, ir.Arg)):
                # Position globals and arguments at the top of the graph.
                kwargs['rank'] = str(0)
            draw_stmt(g, stmt, style='rounded', shape='box', **kwargs)

        # Draw remaining uses
        usepoints = ((k, v) for k, vs in self._usepoints.items() for v in vs)
        for k, stmt in usepoints:
            if stmt not in delay_buf:
                draw_stmt(g, stmt, style='rounded', shape='box', color='gray')
                g.edge(nodeid(k), nodeid(stmt), color='gray')

        # View
        return g.render(view=view)


    def get_leaves(self):
        """Returns all vars that are leaf nodes in the graph.
        They are the last values in the dependency chains.
        """
        # Find values that no other value depends on
        tmp = set(self._defpoints)
        for vs in self._depmap.values():
            tmp -= vs
        return tmp
