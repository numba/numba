import timeit
from dataclasses import dataclass
from contextlib import contextmanager

from numba.tests.support import TestCase


@dataclass(frozen=True)
class BenchmarkSegment:
    name: str
    repeat: int
    run_count: int
    duration: float
    average: float

    def show(self) -> str:
        units = ["s", "ms", "us", "ns"]
        for i, unit in enumerate(units):
            time = self.average * 10 ** (i * 3)
            if time >= 1:
                break
        return (f"{self.name:25}: {self.duration:>5.3}s / "
                f"{self.run_count:5} runs = {time:.2f}{unit}")


class PerfTestCase(TestCase):
    _benchmark_repeat_count = 4
    __perf_segments: list[BenchmarkSegment]

    def _callTestMethod(self, method):
        """Override TestCase._callTestMethod to include logic for reporting
        benchmark results.
        """
        self.__perf_segments = []

        super()._callTestMethod(method)
        result = self._outcome.result
        result.benchmarks = self.__perf_segments
        del self.__perf_segments

    @contextmanager
    def benchmark(self, function, name=""):
        """
        Calls `function` in repeat to benchmark.
        Uses timeit under-the-hood.
        If `name` is falsey, uses `function.__name__` as the benchmark name.
        """
        timer = timeit.Timer(function)
        # Autorange will find the run_count needed for at least 0.2 seconds.
        run_count, duration = timer.autorange()
        # Avoid spending too much time. If a benchmark takes more than 3-sec,
        # only repeat once. Otherwise, repeat 3 times for find the best result.
        repeat = 3 if duration < 3 else 1
        results = timer.repeat(repeat=repeat, number=run_count)
        best = min(results)
        seg = BenchmarkSegment(
            name=name or function.__name__,
            duration=best,
            average=best / run_count,
            run_count=run_count,
            repeat=repeat,
        )
        self.__perf_segments.append(seg)
