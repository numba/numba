from typing import Optional
from collections import defaultdict
from abc import ABC, abstractmethod
import atexit
from functools import cache

from numba.core import ir


try:
    import coverage
except ImportError:
    coverage_available = False
else:
    coverage_available = True


def get_active_coverage() -> Optional["coverage.Coverage"]:
    """Get active coverage instance or return None if not found.
    """
    cov = None
    if coverage_available:
        cov = coverage.Coverage.current()
    return cov


@cache
def _get_coverage_data():
    # Make a singleton CoverageData.
    # Avoid writing to disk. Other processes can corrupt the file.
    covdata = coverage.CoverageData(no_disk=True)
    cov = get_active_coverage()
    assert cov is not None, "no active Coverage instance"

    @atexit.register
    def _finalize():
        cov.get_data().update(covdata)

    return covdata


class NotifyCoverageBase(ABC):
    def __init__(self):
        self._covdata = _get_coverage_data()
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
        if loc.filename.endswith(".py"):
            self._arcs_data[loc.filename].add((loc.line, loc.line))

    def close(self):
        covdata = self._covdata
        with covdata._lock:
            covdata.set_context("numba_compiled")
            covdata.add_arcs(self._arcs_data)
