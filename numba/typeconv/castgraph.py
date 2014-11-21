from __future__ import print_function, absolute_import
from collections import defaultdict
from numba.utils import total_ordering


@total_ordering
class _RelValue(object):
    """Ordered categorical value
    """

    def __init__(self, name, precedence):
        self._name = name
        self._precedence = precedence

    def __repr__(self):
        return self._name

    def __lt__(self, other):
        if isinstance(other, _RelValue):
            return self._precedence < other._precedence
        else:
            return NotImplemented

    def __eq__(self, other):
        if isinstance(other, _RelValue):
            return self._precedence == other._precedence
        else:
            return NotImplemented


# Define the casting relations
Nil = _RelValue('Nil', 0)
Promote = _RelValue('Promote', 1)
Safe = _RelValue('Safe', 2)
Unsafe = _RelValue('Unsafe', 3)


class CastSet(object):
    """A set of casting rules.
    """

    def __init__(self):
        self._rels = defaultdict(lambda: Nil)

    def promote(self, to):
        self._rels[to] = Promote
        return True

    def safe(self, to):
        self._rels[to] = Safe
        return True

    def unsafe(self, to):
        self._rels[to] = Unsafe
        return True

    def insert(self, to, rel):
        setrel = min(rel, self._rels[to])
        if setrel is Nil:
            setrel = rel

        old = self._rels[to]
        self._rels[to] = setrel
        return old != setrel

    def items(self):
        return self._rels.items()

    def get(self, item):
        return self._rels[item]

    def __len__(self):
        return len(self._rels)

    def __repr__(self):
        body = ["{rel}({ty})".format(rel=rel, ty=ty)
                for ty, rel in self._rels.items()]
        return "{" + ', '.join(body) + "}"

    def __contains__(self, item):
        return item in self._rels

    def __iter__(self):
        return iter(self._rels.keys())

    def __getitem__(self, item):
        return self._rels[item]


class TypeGraph(object):
    """A graph that maintains the casting relationship of all types.

    This simplifies the definition of casting rules by automatically
    propagating the rules.
    """

    def __init__(self):
        self._forwards = defaultdict(CastSet)
        self._backwards = defaultdict(set)
        self._newrules = []

    def get(self, ty):
        return self._forwards[ty]

    def propagate(self, a, b, baserel):
        backset = self._backwards[a]

        # Forward propagate the relationship to all nodes that b leads to
        for child in self._forwards[b]:
            rel = max(baserel, self._forwards[b][child])
            if a != child:
                if self._forwards[a].insert(child, rel):
                    self._newrules.append((a, child, rel))
                self._backwards[child].add(a)

            # Propagate the relationship from nodes that connects to a
            for backnode in backset:
                if backnode != child:
                    backrel = max(rel, self._forwards[backnode][a])
                    if self._forwards[backnode].insert(child, backrel):
                        self._newrules.append((backnode, child, backrel))
                    self._backwards[child].add(backnode)

        # Every node that leads to a connects to b
        for child in self._backwards[a]:
            rel = max(baserel, self._forwards[child][a])
            if b != child:
                if self._forwards[child].insert(b, rel):
                    self._newrules.append((child, b, rel))
                self._backwards[b].add(child)

    def promote(self, a, b):
        if self._forwards[a].promote(b):
            self._newrules.append((a, b, Promote))
        self._backwards[b].add(a)
        self.propagate(a, b, Promote)

    def safe(self, a, b):
        if self._forwards[a].safe(b):
            self._newrules.append((a, b, Safe))
        self._backwards[b].add(a)
        self.propagate(a, b, Safe)

    def unsafe(self, a, b):
        if self._forwards[a].unsafe(b):
            self._newrules.append((a, b, Unsafe))
        self._backwards[b].add(a)
        self.propagate(a, b, Unsafe)

    def clear(self):
        self._newrules.clear()

    def get_updates(self):
        return iter(self._newrules)
