from typing import Optional
from collections import defaultdict
from abc import ABC, abstractmethod
import atexit

from numba.core import ir


try:
    import coverage
except ImportError:
    coverage_available = False
else:
    coverage_available = True


def get_active_coverage() -> Optional["coverage.Coverage"]:
    cov = None
    if coverage_available:
        cov = coverage.Coverage.current()
    return cov


class NotifyCoverageBase(ABC):
    def __init__(self, cov=None):
        self._cov = cov or get_active_coverage()
        if self._cov is None:
            raise RuntimeError("no active Coverage object")
        self._init()

    def _init(self):
        pass

    @abstractmethod
    def notify(self, loc: ir.Loc) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class NotifyCompilerCoverage(NotifyCoverageBase):
    """
    Use to notify coverage about compiled lines.
    """
    def _init(self):
        super()._init()
        self._arcs_data = defaultdict(set)

    def notify(self, loc: ir.Loc):
        self._arcs_data[loc.filename].add((loc.line, loc.line))

    def close(self):
        # Avoid writing to disk. Other processes can corrupt the file.
        covdata = coverage.CoverageData(no_disk=True)
        covdata.set_context("numba_compiled")
        covdata.add_arcs(self._arcs_data)

        @todos.append
        def finalize():
            curdata = self._cov.get_data()
            curdata.update(covdata)


todos = []


@atexit.register
def finalize():
    for todo in todos:
        todo()
