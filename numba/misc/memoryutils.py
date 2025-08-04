"""
Memory monitoring utilities for testing.
"""
import io
import os
import sys
import atexit

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


def get_memory_usage():
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
            memory_info['rss'] = mem_info.rss
            memory_info['vms'] = mem_info.vms

            # Get system memory info
            sys_mem = psutil.virtual_memory()
            memory_info['available'] = sys_mem.available
            memory_info['total'] = sys_mem.total
            memory_info['percent_used'] = sys_mem.percent

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # Fallback to resource module if psutil fails
            pass

    # Use resource module as fallback or supplement (not available on Windows)
    if _HAS_RESOURCE:
        try:
            # Get resource usage
            usage = resource.getrusage(resource.RUSAGE_SELF)

            # On Linux, ru_maxrss is in KB, on macOS it's in bytes
            if sys.platform == 'darwin':
                memory_info['peak_rss'] = usage.ru_maxrss  # bytes
            else:
                memory_info['peak_rss'] = usage.ru_maxrss * 1024  # KB to bytes

            # If psutil is not available, use resource module values
            if not _HAS_PSUTIL:
                memory_info['rss'] = memory_info['peak_rss']
                memory_info['vms'] = memory_info['peak_rss']  # Approximation

        except (OSError, AttributeError):
            # resource module doesn't support getrusage
            pass

    # If neither psutil nor resource is available (e.g., on Windows),
    # provide minimal fallback information
    if not _HAS_PSUTIL and not _HAS_RESOURCE:
        # On Windows without psutil, we have limited options
        # We can't get accurate memory information without external libraries
        memory_info['rss'] = None
        memory_info['vms'] = None
        memory_info['peak_rss'] = None

    return memory_info


def format_memory_usage(memory_info):
    """
    Format memory usage information for display.

    Args:
        memory_info (dict): Memory info from get_memory_usage()

    Returns:
        str: Formatted memory usage string
    """
    def format_bytes(bytes_val):
        """Convert bytes to human readable format"""
        if bytes_val is None:
            return "N/A"

        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_val < 1024.0:
                return f"{bytes_val:.2f} {unit}"
            bytes_val /= 1024.0
        return f"{bytes_val:.2f} TB"

    parts = []

    if 'rss' in memory_info:
        parts.append(f"RSS: {format_bytes(memory_info['rss'])}")

    if 'vms' in memory_info:
        parts.append(f"VMS: {format_bytes(memory_info['vms'])}")

    if 'peak_rss' in memory_info:
        parts.append(f"Peak RSS: {format_bytes(memory_info['peak_rss'])}")

    if 'available' in memory_info:
        parts.append(f"Available: {format_bytes(memory_info['available'])}")

    if 'percent_used' in memory_info:
        parts.append(f"Used: {memory_info['percent_used']:.1f}%")

    return " | ".join(parts) if parts else "Memory info unavailable"


def monitor_memory_usage():
    """
    Get and format current memory usage for display.

    Returns:
        str: Formatted memory usage string
    """
    memory_info = get_memory_usage()
    return format_memory_usage(memory_info)

# Additional imports for context manager
import contextlib
import time
from collections import defaultdict

# Global storage for memory monitoring data
_memory_records = []
_memory_stats = {}


class MemoryRecord:
    """A record of memory usage during a test or operation."""

    def __init__(self, name, start_memory, end_memory, duration):
        self.name = name
        self.start_memory = start_memory
        self.end_memory = end_memory
        self.duration = duration
        self.timestamp = time.time()

    def get_memory_delta(self):
        """Calculate memory usage delta between start and end."""
        delta = {}

        for key in ['rss', 'vms', 'peak_rss']:
            start_val = self.start_memory.get(key, 0)
            end_val = self.end_memory.get(key, 0)
            if start_val and end_val:
                delta[key] = end_val - start_val

        return delta

    def format_record(self):
        """Format the memory record for display."""
        delta = self.get_memory_delta()

        def format_bytes(bytes_val):
            """Convert bytes to human readable format"""
            if bytes_val is None or bytes_val == 0:
                return "0 B"

            sign = "-" if bytes_val < 0 else "+"
            bytes_val = abs(bytes_val)

            for unit in ['B', 'KB', 'MB', 'GB']:
                if bytes_val < 1024.0:
                    return f"{sign}{bytes_val:.2f} {unit}"
                bytes_val /= 1024.0
            return f"{sign}{bytes_val:.2f} TB"

        parts = [f"Test: {self.name}"]
        parts.append(f"Duration: {self.duration:.3f}s")

        if 'rss' in delta:
            parts.append(f"RSS Δ: {format_bytes(delta['rss'])}")

        if 'vms' in delta:
            parts.append(f"VMS Δ: {format_bytes(delta['vms'])}")

        if 'peak_rss' in delta:
            parts.append(f"Peak RSS Δ: {format_bytes(delta['peak_rss'])}")

        return " | ".join(parts)


_atexit_installed = False


def install_atexit(filename) -> None:
    global _atexit_installed

    if _atexit_installed:
        return

    _atexit_installed = True

    def memory_write_handler():
        print(get_memory_log())

    atexit.register(memory_write_handler)


def get_memory_log():
    with io.StringIO() as f:
        _write_memory_log(f, _memory_records)
        return f.getvalue()


@contextlib.contextmanager
def memory_monitor(name):
    """
    Context manager to monitor memory usage during test execution.

    Args:
        name (str): Name/identifier for the test or operation being monitored

    Yields:
        MemoryRecord: The memory record that will be populated during execution
    """

    # Record start time and memory usage
    start_time = time.time()
    start_memory = get_memory_usage()

    # Create a record object that will be populated
    record = None

    try:
        yield record
    finally:
        # Record end time and memory usage
        end_time = time.time()
        end_memory = get_memory_usage()
        duration = end_time - start_time

        # Create the memory record
        record = MemoryRecord(name, start_memory, end_memory, duration)

        # Store the record globally for later retrieval
        _memory_records.append(record)

        # Also store in categorized stats
        if name not in _memory_stats:
            _memory_stats[name] = []
        _memory_stats[name].append(record)


def get_memory_records():
    """
    Get all memory records collected so far.

    Returns:
        list: List of MemoryRecord objects
    """
    return _memory_records.copy()


def get_memory_stats():
    """
    Get memory statistics organized by test name.

    Returns:
        dict: Dictionary mapping test names to lists of MemoryRecord objects
    """
    return dict(_memory_stats)


def clear_memory_records():
    """Clear all stored memory records."""
    global _memory_records, _memory_stats
    _memory_records.clear()
    _memory_stats.clear()


def write_memory_log(filename, records=None):
    """
    Write memory monitoring data to a log file.

    Args:
        filename (str): Path to the log file
        records (list, optional): List of MemoryRecord objects to write.
                                 If None, uses all stored records.
    """
    if records is None:
        records = _memory_records

    with open(filename, 'w') as f:
        _write_memory_log(f, records)


def _write_memory_log(f, records):
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
    total_rss_delta = sum(r.get_memory_delta().get('rss', 0) for r in records)
    total_vms_delta = sum(r.get_memory_delta().get('vms', 0) for r in records)

    def format_bytes(bytes_val):
        if bytes_val == 0:
            return "0 B"
        sign = "-" if bytes_val < 0 else "+"
        bytes_val = abs(bytes_val)
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_val < 1024.0:
                return f"{sign}{bytes_val:.2f} {unit}"
            bytes_val /= 1024.0
        return f"{sign}{bytes_val:.2f} TB"

    f.write(f"Total RSS change: {format_bytes(total_rss_delta)}\n")
    f.write(f"Total VMS change: {format_bytes(total_vms_delta)}\n")
    f.write(f"Average test duration: {sum(r.duration for r in records) / len(records):.3f}s\n")
    f.write("\n")

    # Sort records by memory use increase (RSS delta) in descending order
    sorted_records = sorted(records,
                            key=lambda r: r.get_memory_delta().get('rss', 0),
                            reverse=True)

    # Write individual records
    f.write("Individual Test Records (sorted by memory increase):\n")
    f.write("-" * 50 + "\n")

    for record in sorted_records:
        f.write(f"{record.format_record()}\n")

    f.write("\n")
    f.write("Log generated at: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")


def format_memory_summary():
    """
    Format a summary of all memory records for display.

    Returns:
        str: Formatted summary string
    """
    if not _memory_records:
        return "No memory records available"

    total_tests = len(_memory_records)
    total_rss_delta = sum(r.get_memory_delta().get('rss', 0) for r in _memory_records)
    total_duration = sum(r.duration for r in _memory_records)

    def format_bytes(bytes_val):
        if bytes_val == 0:
            return "0 B"
        sign = "-" if bytes_val < 0 else "+"
        bytes_val = abs(bytes_val)
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_val < 1024.0:
                return f"{sign}{bytes_val:.2f} {unit}"
            bytes_val /= 1024.0
        return f"{sign}{bytes_val:.2f} TB"

    return (f"Memory Summary: {total_tests} tests, "
            f"Total RSS Δ: {format_bytes(total_rss_delta)}, "
            f"Total Duration: {total_duration:.3f}s")