from collections import defaultdict
from numba.utils import total_ordering


@total_ordering
class _RelValue(object):
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


Nil = _RelValue('Nil', 0)
Promote = _RelValue('Promote', 1)
Safe = _RelValue('Safe', 2)
Unsafe = _RelValue('Unsafe', 3)


class CastSet(object):
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

    def __repr__(self):
        body = ["{rel}({ty})".format(rel=rel, ty=ty)
                for ty, rel in self._rels.items()]
        return "{" + ', '.join(body) + "}"

    def __contains__(self, item):
        return item in self._rels

    def __iter__(self):
        return iter(self._rels.keys())

    def items(self):
        return self._rels.items()

    def __getitem__(self, item):
        return self._rels[item]

    def get(self, item):
        return self._rels[item]


class TypeGraph(object):
    def __init__(self):
        self._forwards = defaultdict(CastSet)
        self._backwards = defaultdict(set)
        self._newrules = []

    def get(self, ty):
        return self._forwards[ty]

    def propagate(self, a, b, baserel):
        # print("++", baserel, a, '->', b)
        backset = self._backwards[a]

        # Forward propagate the relationship to all nodes that b leads to
        for child in self._forwards[b]:
            rel = max(baserel, self._forwards[b][child])
            if a != child:
                # print(" f", rel, a, '->', child)
                if self._forwards[a].insert(child, rel):
                    self._newrules.append((a, child, rel))
                self._backwards[child].add(a)

            # Propagate the relationship from nodes that connects to a
            for backnode in backset:
                if backnode != child:
                    backrel = max(rel, self._forwards[backnode][a])
                    # print(" bf", backrel, backnode, '->', child)
                    if self._forwards[backnode].insert(child, backrel):
                        self._newrules.append((backnode, child, backrel))
                    self._backwards[child].add(backnode)

        # Every node that leads to a connects to b
        for child in self._backwards[a]:
            rel = max(baserel, self._forwards[child][a])
            if b != child:
                # print(" b", rel, child, '->', b)
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

#
# tg = TypeGraph()
# tg.promote('i8', 'i16')
# tg.unsafe('i16', 'i8')
#
# tg.unsafe('i32', 'i16')
# tg.promote('i16', 'i32')
#
# tg.safe('i32', 'f64')
# tg.unsafe('f64', 'f32')
# tg.promote('f32', 'f64')
#
# tg.safe('i16', 'f32')
# tg.unsafe('f32', 'i16')
#
# tg.safe('f32', 'c64')
# tg.unsafe('c64', 'f32')
#
# tg.safe('f64', 'c128')
# tg.unsafe('c128', 'c64')
#
# tg.promote('c64', 'c128')
#
# print(tg.get('i8'))
# print(tg.get('i16'))
# print(tg.get('i32'))
# print(tg.get('f32'))
# print(tg.get('f64'))
# print(tg.get('c64'))
# print(tg.get('c128'))


