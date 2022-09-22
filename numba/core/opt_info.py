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

    def needs_debug_info(self) -> bool:
        """
        Determines if the processor requires debugging information (file and
        line numbers) or not.

        The default is to require it, but, if overridden to false, this can
        save time at compilation.
        """
        return True

    def process(self, remarks_data: List[Any], full_name: str) ->\
            Iterable[Tuple[str, Any]]:
        """
        Generate optimisation information from the remarks.

        Parameters
        ----------
        remarks_data : a list of the parsed YAML provided by LLVM; there may
        be multiple compilation passes and remarks will be generated for
        each pass. Note that this data is shared and must not be mutated.
        full_name : the complete Numba name for the function being compiled

        Returns
        -------
        A collection of key-value pairs with the key as a unique name and the
        value as the optimisation information.
        """
        pass


class Aggregate(OptimizationProcessor):
    """
    A processor which is a collection of other processors
    """
    def __init__(self, *processors):
        self._processors = processors

    def filters(self) -> Iterable[str]:
        return (f for p in self._processors for f in p.filters())

    def needs_debug_info(self) -> bool:
        return any(p.needs_debug_info() for p in self._processors)

    def process(self, remarks_data: List[Any], full_name: str) ->\
            Iterable[Tuple[str, Any]]:
        return (o for p in self._processors for o in p.process(remarks_data,
                                                               full_name))


class RawOptimizationRemarks(OptimizationProcessor):
    """
    This stores raw optimisation remarks

    It is almost certainly a bad idea to use as the remarks are large and
    inefficient. This is intended as a utility for viewing remarks
    information when developing new processors.
    """

    def __init__(self, *filter_names):
        self.filter_names = set(filter_names)

    def filters(self) -> Iterable[str]:
        return self.filter_names

    def needs_debug_info(self) -> bool:
        return False

    def process(self, remarks_data: List[Any], function) -> Iterable[
          Tuple[str, Any]]:
        return ('raw', remarks_data),


class LoopDeLoop(OptimizationProcessor):
    """
    Indicates whether a loop vectorization happened.

    Vectorization replaces loops with CPU instructions that can do operations in
    parallel. The resulting loop will be much faster. Not all loops can be
    vectorized as the set of CPU instructions is limited to "pure arithmetic"
    operations.

    The output of this processor will be a dictionary with a (file, line,
    column) tuple and a true if the loop was vectorized; false otherwise.
    """

    def filters(self) -> Iterable[str]:
        return 'loop-vectorize',

    def process(self, remarks_data: List[Any], function) -> Iterable[
           Tuple[str, Any]]:
        loops = {}
        for remarks in remarks_data:
            for entry in remarks:
                if isinstance(entry, (Missed, Passed)) and entry.info['Pass']\
                        == 'loop-vectorize' and 'DebugLoc' in entry.info:
                    location = (entry.info['DebugLoc']['File'],
                                entry.info['DebugLoc']['Line'],
                                entry.info['DebugLoc']['Column'])
                    loops[location] = (loops.get(location, False) or
                                       isinstance(entry, Passed))
        return ('loop_vectorization', loops),


class SuperWorldLevelParallelismDetector(OptimizationProcessor):
    """
    Determines locations where the SLP detector found a way to turn scalar
    operations into vector operations.

    CPUs have some operations that are meant for doing arithmetic operations
    over vectors quickly. Sometimes, even though code is being done over scalar
    values, it has a structure that looks like a vector, so it can be more
    efficient to convert the data into a vector and use the special operations.

    The output of this processor is a set of (file, line, column) tuples where
    scalar operations were successfully converted to a faster vector operation.

    https://llvm.org/docs/Vectorizers.html#the-slp-vectorizer
    """

    def filters(self) -> Iterable[str]:
        return 'slp-vectorizer',

    def process(self, remarks_data: List[Any], function) -> Iterable[
           Tuple[str, Any]]:
        slps = set()
        for remarks in remarks_data:
            for entry in remarks:
                if isinstance(entry,  Passed) and entry.info['Pass']\
                        == 'slp-vectorizer' and 'DebugLoc' in entry.info:
                    slps.add((entry.info['DebugLoc']['File'],
                              entry.info['DebugLoc']['Line'],
                              entry.info['DebugLoc']['Column']))
        return ('slp_vectorization', slps),


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

Analysis = namedtuple("Analysis", ("info", ))

AnalysisFPCommute = namedtuple("AnalysisFPCommute", ("info", ))

AnalysisAliasing = namedtuple("AnalysisAliasing", ("info", ))

Failure = namedtuple("Failure", ("info", ))


def global_processors() -> Iterable[OptimizationProcessor]:
    """
    Iterate over all globally registered optimisation processors
    """
    return _global_processors
