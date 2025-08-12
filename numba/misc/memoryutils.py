"""
Memory monitoring utilities
"""
from __future__ import annotations
import io
import os
import atexit
import pickle
import contextlib
import time
from typing import Dict, List, Optional, TextIO
from numba.core.config import IS_OSX

# resource module is not available on Windows
try:
    import resource

    _HAS_RESOURCE = True
except ImportError:
    _HAS_RESOURCE = False

try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def _get_memory_usage() -> Dict[str, Optional[int]]:
    """
    Get current memory usage information.

    Returns:
        dict: Memory usage information including:
            - rss: Resident Set Size (physical memory currently used)
            - vms: Virtual Memory Size (virtual memory used)
            - peak_rss: Peak RSS usage
            - available: Available memory (if psutil is available)
    """
    memory_info = {}

    if _HAS_PSUTIL:
        try:
            # Get current process
            process = psutil.Process(os.getpid())

            # Get memory info
            mem_info = process.memory_info()
            memory_info["rss"] = mem_info.rss
            memory_info["vms"] = mem_info.vms

            # Get system memory info
            sys_mem = psutil.virtual_memory()
            memory_info["available"] = sys_mem.available
            memory_info["total"] = sys_mem.total
            memory_info["percent_used"] = sys_mem.percent

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # Fallback to resource module if psutil fails
            pass

    # Use resource module as fallback or supplement (not available on Windows)
    if _HAS_RESOURCE:
        try:
            # Get resource usage
            usage = resource.getrusage(resource.RUSAGE_SELF)

            # On Linux, ru_maxrss is in KB, on macOS it's in bytes
            if IS_OSX:
                memory_info["peak_rss"] = usage.ru_maxrss  # bytes
            else:
                memory_info["peak_rss"] = usage.ru_maxrss * 1024  # KB to bytes

            # If psutil is not available, use resource module values
            if not _HAS_PSUTIL:
                memory_info["rss"] = memory_info["peak_rss"]
                memory_info["vms"] = memory_info["peak_rss"]  # Approximation

        except (OSError, AttributeError):
            # resource module doesn't support getrusage
            pass

    # If neither psutil nor resource is available (e.g., on Windows),
    # provide minimal fallback information
    if not _HAS_PSUTIL and not _HAS_RESOURCE:
        # On Windows without psutil, we have limited options
        # We can't get accurate memory information without external libraries
        memory_info["rss"] = None
        memory_info["vms"] = None
        memory_info["peak_rss"] = None

    return memory_info


# Global storage for memory monitoring data
_memory_records: list[_MemoryRecord] = []


class _MemoryRecord:
    """A record of memory usage during a test or operation."""

    def __init__(
        self,
        name: str,
        start_memory: Dict[str, Optional[int]],
        end_memory: Dict[str, Optional[int]],
        duration: float,
    ) -> None:
        self.name = name
        self.start_memory = start_memory
        self.end_memory = end_memory
        self.duration = duration
        self.timestamp = time.time()

    def get_memory_delta(self) -> Dict[str, int]:
        """Calculate memory usage delta between start and end."""
        delta = {}

        for key in ["rss", "vms", "peak_rss"]:
            start_val = self.start_memory.get(key, 0)
            end_val = self.end_memory.get(key, 0)
            if start_val and end_val:
                delta[key] = end_val - start_val

        return delta

    def format_record(self) -> str:
        """Format the memory record for display."""
        delta = self.get_memory_delta()

        def format_bytes(bytes_val: Optional[float]) -> str:
            """Convert bytes to human readable format"""
            if bytes_val is None or bytes_val == 0:
                return "0 B"

            sign = "-" if bytes_val < 0 else "+"
            bytes_val = abs(bytes_val)

            for unit in ["B", "KB", "MB", "GB"]:
                if bytes_val < 1024.0:
                    return f"{sign}{bytes_val:.2f} {unit}"
                bytes_val /= 1024.0
            return f"{sign}{bytes_val:.2f} TB"

        parts = [f"Test: {self.name}"]
        parts.append(f"Duration: {self.duration:.3f}s")

        if "rss" in delta:
            parts.append(f"RSS delta: {format_bytes(delta['rss'])}")

        if "vms" in delta:
            parts.append(f"VMS delta: {format_bytes(delta['vms'])}")

        if "peak_rss" in delta:
            parts.append(f"Peak RSS delta: {format_bytes(delta['peak_rss'])}")

        return " | ".join(parts)


_atexit_installed = False


def install_atexit(filename) -> None:
    global _atexit_installed

    if _atexit_installed:
        return

    _atexit_installed = True

    def memory_write_handler():
        summary = get_memory_log(_memory_records, topk=-1)
        print(summary)
        with open(filename, "ab") as fout:
            pickle.dump(
                _memory_records, fout, protocol=pickle.HIGHEST_PROTOCOL
            )

    atexit.register(memory_write_handler)


def get_memory_log(records: List[_MemoryRecord], topk=20) -> str:
    with io.StringIO() as f:
        _write_memory_log(f, records, topk=topk)
        return f.getvalue()


@contextlib.contextmanager
def memory_monitor(name: str):
    """
    Context manager to monitor memory usage during test execution.

    Args:
        name (str): Name/identifier for the test or operation being monitored
    """

    # Record start time and memory usage
    start_time = time.time()
    start_memory = _get_memory_usage()

    try:
        yield
    finally:
        # Record end time and memory usage
        end_time = time.time()
        end_memory = _get_memory_usage()
        duration = end_time - start_time

        # Create the memory record
        record = _MemoryRecord(name, start_memory, end_memory, duration)

        # Store the record globally for later retrieval
        _memory_records.append(record)


def _write_memory_log(
    f: TextIO, records: List[_MemoryRecord], topk: int = 20
) -> None:
    f.write("Memory Usage Log\n")
    f.write("=" * 50 + "\n\n")

    if not records:
        f.write("No memory records found.\n")
        return

    # Write summary statistics
    f.write("Summary Statistics:\n")
    f.write("-" * 20 + "\n")
    f.write(f"Total tests monitored: {len(records)}\n")

    # Calculate total memory change
    total_rss_delta = sum(r.get_memory_delta().get("rss", 0) for r in records)
    total_vms_delta = sum(r.get_memory_delta().get("vms", 0) for r in records)

    def format_bytes(bytes_val: float) -> str:
        if bytes_val == 0:
            return "0 B"
        sign = "-" if bytes_val < 0 else "+"
        bytes_val = abs(bytes_val)
        for unit in ["B", "KB", "MB", "GB"]:
            if bytes_val < 1024.0:
                return f"{sign}{bytes_val:.2f} {unit}"
            bytes_val /= 1024.0
        return f"{sign}{bytes_val:.2f} TB"

    f.write(f"Total RSS change: {format_bytes(total_rss_delta)}\n")
    f.write(f"Total VMS change: {format_bytes(total_vms_delta)}\n")
    f.write(
        f"Average test duration: "
        f"{sum(r.duration for r in records) / len(records):.3f}s\n"
    )
    f.write("\n")

    # Sort records by memory use increase (RSS delta) in descending order
    sorted_records = sorted(
        records, key=lambda r: r.get_memory_delta().get("rss", 0), reverse=True
    )

    # Write individual records
    f.write("Individual Test Records (descending order by RSS delta):\n")
    f.write("-" * 50 + "\n")

    # Show only top k records
    top_records = sorted_records[:topk]
    for record in top_records:
        f.write(f"{record.format_record()}\n")

    f.write("\n")
    f.write("Log generated at: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
