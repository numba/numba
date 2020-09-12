import enum
import typing as pt
from collections import defaultdict

from numba.core.types import Type


class Conversion(enum.IntEnum):
    """
    A conversion kind from one type to the other.  The enum members
    are ordered from stricter to looser.
    """
    # The two types are identical
    exact = 1
    # The two types are of the same kind, the destination type has more
    # extension or precision than the source type (e.g. float32 -> float64,
    # or int32 -> int64)
    promote = 2
    # The source type can be converted to the destination type without loss
    # of information (e.g. int32 -> int64).  Note that the conversion may
    # still fail explicitly at runtime (e.g. Optional(int32) -> int32)
    safe = 3
    # The conversion may appear to succeed at runtime while losing information
    # or precision (e.g. int32 -> uint32, float64 -> float32, int64 -> int32,
    # etc.)
    unsafe = 4

    # This value is only used internally
    nil = 99


class CastSet(object):
    """A set of casting rules.

    There is at most one rule per target type.
    """

    def __init__(self) -> None:
        self._rels: pt.Dict[Type, Conversion] = {}

    def insert(self, to: Type, rel: Conversion) -> bool:
        old = self.get(to)
        setrel = min(rel, old)
        self._rels[to] = setrel
        return old != setrel

    def items(self) -> pt.Iterable[pt.Tuple[Type, Conversion]]:
        return self._rels.items()

    def get(self, item: Type) -> Conversion:
        return self._rels.get(item, Conversion.nil)

    def __len__(self) -> int:
        return len(self._rels)

    def __repr__(self) -> str:
        body = ["{rel}({ty})".format(rel=rel, ty=ty)
                for ty, rel in self._rels.items()]
        return "{" + ', '.join(body) + "}"

    def __contains__(self, item: Type) -> bool:
        return item in self._rels

    def __iter__(self) -> pt.Iterator[Type]:
        return iter(self._rels.keys())

    def __getitem__(self, item: Type) -> Conversion:
        return self._rels[item]


class TypeGraph(object):
    """A graph that maintains the casting relationship of all types.

    This simplifies the definition of casting rules by automatically
    propagating the rules.
    """

    def __init__(
        self,
        callback: pt.Callable[[Type, Type, Conversion], None],
    ):
        """
        Args
        ----
        - callback: callable
            It is called for each new casting rule with
            (from_type, to_type, castrel).
        """
        assert callback is None or callable(callback)
        self._forwards: pt.DefaultDict[Type, CastSet] = defaultdict(CastSet)
        self._backwards: pt.DefaultDict[Type, pt.Set[Type]] = defaultdict(set)
        self._callback = callback

    def get(self, ty: Type) -> CastSet:
        return self._forwards[ty]

    def propagate(self, a: Type, b: Type, baserel: Conversion) -> None:
        backset = self._backwards[a]

        # Forward propagate the relationship to all nodes that b leads to
        for child in self._forwards[b]:
            rel = max(baserel, self._forwards[b][child])
            if a != child:
                if self._forwards[a].insert(child, rel):
                    self._callback(a, child, rel)
                self._backwards[child].add(a)

            # Propagate the relationship from nodes that connects to a
            for backnode in backset:
                if backnode != child:
                    backrel = max(rel, self._forwards[backnode][a])
                    if self._forwards[backnode].insert(child, backrel):
                        self._callback(backnode, child, backrel)
                    self._backwards[child].add(backnode)

        # Every node that leads to a connects to b
        for child in self._backwards[a]:
            rel = max(baserel, self._forwards[child][a])
            if b != child:
                if self._forwards[child].insert(b, rel):
                    self._callback(child, b, rel)
                self._backwards[b].add(child)

    def insert_rule(self, a: Type, b: Type, rel: Conversion) -> None:
        self._forwards[a].insert(b, rel)
        self._callback(a, b, rel)
        self._backwards[b].add(a)
        self.propagate(a, b, rel)

    def promote(self, a: Type, b: Type) -> None:
        self.insert_rule(a, b, Conversion.promote)

    def safe(self, a: Type, b: Type) -> None:
        self.insert_rule(a, b, Conversion.safe)

    def unsafe(self, a: Type, b: Type) -> None:
        self.insert_rule(a, b, Conversion.unsafe)
