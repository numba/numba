import re
import itertools
import operator
import heapq
from collections import namedtuple
from collections.abc import Sequence
from contextlib import contextmanager

from numba.core.utils import cached_property

import llvmlite.binding as llvm


class RecordLLVMPassTimings:
    """A helper context manager to track LLVM pass timings.
    """

    __slots__ = ["_data"]

    def __enter__(self):
        """Enables the pass timing in LLVM.
        """
        llvm.enable_time_passes()
        return self

    def __exit__(self, exc_val, exc_type, exc_tb):
        """Reset timings and save report internally.
        """
        self._data = llvm.report_and_reset_timings()
        return

    def get(self):
        """Retrieve timing data for processing.

        Returns
        -------
        timings : _ProcessedPassTimings
        """
        return _ProcessedPassTimings(self._data)


_PassTimingRecord = namedtuple(
    "_PassTimingRecord",
    [
        "user_time",
        "user_percent",
        "system_time",
        "system_percent",
        "user_system_time",
        "user_system_percent",
        "wall_time",
        "wall_percent",
        "pass_name",
    ],
)


def _adjust_timings(records):
    """Adjust timing records because of truncated information.

    Details: The percent information can be used to improve the timing
    information.

    Returns
    -------
    res : List[_PassTimingRecord]
    """
    total_rec = records[-1]
    assert total_rec.pass_name == "Total"  # guard for implementation error

    def make_adjuster(attr):
        time_attr = f"{attr}_time"
        percent_attr = f"{attr}_percent"
        time_getter = operator.attrgetter(time_attr)

        def adjust(d):
            """Compute percent x total_time = adjusted"""
            total = time_getter(total_rec)
            adjusted = total * d[percent_attr] * 0.01
            d[time_attr] = adjusted
            return d

        return adjust

    # Make adjustment functions for each field
    adj_fns = [
        make_adjuster(x) for x in ["user", "system", "user_system", "wall"]
    ]

    # Extract dictionaries from the namedtuples
    dicts = map(lambda x: x._asdict(), records)

    def chained(d):
        # Chain the adjustment functions
        for fn in adj_fns:
            d = fn(d)
        # Reconstruct the namedtuple
        return _PassTimingRecord(**d)

    return list(map(chained, dicts))


class _ProcessedPassTimings:
    """A class for processing and grouping raw timing data from LLVM.

    The processing is done lazily so we don't waste time processing unused
    timing information.

    The per-pass timings are grouped because LLVM may sometime print multiple
    timing report.
    """

    def __init__(self, raw_data):
        self._raw_data = raw_data

    def __bool__(self):
        return bool(self._raw_data)

    def get_raw_data(self):
        """Returns the raw string data
        """
        return self._raw_data

    def get_total_time(self):
        """Compute the total time spend in all pass-groups.
        """
        return sum(grp[-1].wall_time for grp in self.list_pass_groups())

    def list_pass_groups(self):
        """Get the processed data for all pass-groups.

        Returns
        -------
        res : List[List[_PassTimingRecord]]
        """
        return self._processed

    def list_tops(self, n):
        """Returns the top(n) most time-consuming (by wall-time) passes for
        each group.

        Parameters
        ----------
        n : int
            This limits the maximum number of items to show.
            This function will show the ``n`` most time-consuming passes.

        Returns
        -------
        res : List[List[_PassTimingRecord]]
            Returns the top(n) most time-consuming passes in descending order
            for each pass-group.
        """
        key = operator.attrgetter("wall_time")
        return [
            heapq.nlargest(n, grp[:-1], key) for grp in self.list_pass_groups()
        ]

    def summary(self, topn=5):
        """Return a string summarizing the timing information.

        Parameters
        ----------
        topn : int
            This limits the maximum number of items to show per pass group.
            This function will show the ``topn`` most time-consuming passes.

        Returns
        -------
        res : str
        """
        buf = []
        ap = buf.append
        for i, top in enumerate(self.list_tops(topn), 1):
            ap(f"Pass group #{i} took {self.get_total_time():.4f}s")
            ap("  Top timings:")
            for p in top:
                ap(f"  {p.wall_time:.4f}s ({p.wall_percent:5}%) {p.pass_name}")
        return "\n".join(buf)

    @cached_property
    def _processed(self):
        """A cached property for lazily processing the data and returning it.

        See ``_process()`` for details.
        """
        return self._process()

    def _process(self):
        """Parses the raw string data from LLVM timing report and attempts
        to improve the data by recomputing the times
        (See `_adjust_timings()``).
        """

        def parse(raw_data):
            """A generator that parses the raw_data line-by-line to extract
            timing information for each pass.
            """
            lines = raw_data.splitlines()
            n = r"\s*((?:[0-9]+\.)?[0-9]+)"
            pat = f"\\s+{n}\\s*\\({n}%\\)" * 4 + r"\s*(.*)"

            for ln in lines:
                m = re.match(pat, ln)
                if m is not None:
                    raw_data = m.groups()
                    rec = _PassTimingRecord(
                        *map(float, raw_data[:-1]), *raw_data[-1:]
                    )
                    yield rec
                    if rec.pass_name == "Total":
                        # "Total" means a pass group has completed
                        yield None

        # Parse iterator
        parse_iter = iter(parse(self._raw_data))
        runs = []

        def not_none(x):
            return x is not None

        while True:
            # Group the timings by pass group
            grouped = list(itertools.takewhile(not_none, parse_iter))
            if grouped:
                # Save the pass group
                runs.append(_adjust_timings(grouped))
            else:
                # Nothing yielded, parsing is done
                break

        return runs


_NamedTimings = namedtuple("_NamedTimings", ["name", "timings"])


class PassTimingCollection(Sequence):
    """A collection of pass timings.
    """

    def __init__(self, name):
        self._name = name
        self._records = []

    @contextmanager
    def record(self, name):
        """Record timings

        See also ``RecordLLVMPassTimings``

        Parameters
        ----------
        name : str
            Name for the records.
        """
        with RecordLLVMPassTimings() as timings:
            yield
        rec = timings.get()
        # Only keep non-empty records
        if rec:
            self.append(name, rec)

    def append(self, name, timings):
        """Append timing records

        Parameters
        ----------
        name : str
            Name for the records.
        timings : _ProcessedPassTimings
            the timing records.
        """
        self._records.append(_NamedTimings(name, timings))

    def __getitem__(self, i):
        """Get the i-th timing record.

        Returns
        -------
        res : _NamedTimings
        """
        return self._records[i]

    def __len__(self):
        """Length of this collection.
        """
        return len(self._records)

    def __str__(self):
        buf = []
        ap = buf.append
        ap(f"Printing pass timings for {self._name}")
        for r in self._records:
            ap(f"== {r.name}")
            ap(r.timings.summary())
        return "\n".join(buf)
