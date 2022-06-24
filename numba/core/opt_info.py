from typing import Any, Iterable, List, Set, Tuple
from collections import namedtuple


class OptimizationProcessor:
    """
    A converter that reads LLVM optimization remarks and produced some useful
    information for the programmer
    """
    def filters(self) -> Iterable[str]:
        """
        Provide a list of identifiers that will be requested during compilation.
        These can be regular expressions.
        """
        pass

    def process(self, remarks_data: List[Any], full_name: str) ->\
            Iterable[Tuple[str, Any]]:
        """
        Generate optimisation information from the remarks.

        Parameters
        ----------
        remarks_data : a list of the parsed YAML provided by LLVM; there may
        be multiple complilation passes and remarks will be generated for
        each pass. Note that this data is shared and must not be mutated.
        full_name : the complete Numba name for the function being compiled

        Returns
        -------
        A collection of key-value pairs with the key as a unique name and the
        value as the optimisation information.
        """
        pass


class RawOptimizationRemarks(OptimizationProcessor):
    """
    This stores raw optimisation remarks

    It is almost certainly a bad idea to use as the remarks are large and
    inefficient.
    """

    def __init__(self, *filter_names):
        self.filter_names = set(filter_names)

    def filters(self) -> Iterable[str]:
        return self.filter_names

    def process(self, remarks_data: Any, function) -> Iterable[Tuple[str, Any]]:
        return ('raw', remarks_data),


_global_processors: Set[OptimizationProcessor] = set()


def register_processor(processor: OptimizationProcessor):
    """
    Add a processor to the set of processors that is run on every function.

    Parameters
    ----------
    processor : The optimisation processor to add
    """
    _global_processors.add(processor)


Missed = namedtuple("Missed", ("info",))

Passed = namedtuple("Passed", ("info",))


def global_processors() -> Iterable[OptimizationProcessor]:
    """
    Iterate over all globally registered optimisation processors
    """
    return _global_processors
