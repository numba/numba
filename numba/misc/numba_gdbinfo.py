"""Module for displaying information about Numba's gdb set up"""


def _run_cmd(cmdline, env):
    """Runs cmdline (list of string args) in a subprocess with environment dict
    env.
    """
    import subprocess
    import threading
    popen = subprocess.Popen(cmdline,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             env=env)
    # finish in 10 seconds or kill it
    timeout = threading.Timer(10, popen.kill)
    try:
        timeout.start()
        out, err = popen.communicate()
        if popen.returncode != 0:
            raise AssertionError(
                "process failed with code %s: stderr follows\n%s\n" %
                (popen.returncode, err.decode()))
        return out.decode(), err.decode()
    finally:
        timeout.cancel()
    return None, None


def display_gdbinfo(sep_pos=45):
    """Prints information to stdout about the gdb setup that Numba has found"""
    import os
    import re
    from textwrap import dedent
    from numba import config

    print('-' * 80)
    fmt = f'%-{sep_pos}s : %-s'

    print_ext_file = "gdb_print_extension.py"
    print_ext_path = os.path.join(os.path.dirname(__file__), print_ext_file)

    gdb_call = [config.GDB_BINARY, "-q", "-ex",]

    cmd = gdb_call + [("python from __future__ import print_function; "
                       "import sys; print(sys.version_info[:2])"), "-ex", "q"]
    stderr, stdout = _run_cmd(cmd, os.environ)
    version_match = re.match(r'\((\d+),\s+(\d+)\)', stderr.strip())
    if version_match is None:
        gdb_python_version = 'No Python support'
        gdb_python_numpy_version = None
    else:
        pymajor, pyminor = version_match.groups()
        gdb_python_version = f"{pymajor}.{pyminor}"

        cmd = gdb_call + [("python from __future__ import print_function; "
                           "import types; import numpy; "
                           "print(isinstance(numpy, types.ModuleType))"),
                          "-ex", "q"]

        stderr, stdout = _run_cmd(cmd, os.environ)
        if stderr.strip() == 'True':
            # NumPy is present find the version
            cmd = gdb_call + [("python from __future__ import print_function; "
                               "import types; import numpy;"
                               "print(numpy.__version__)"),
                              "-ex", "q"]
            stderr, stdout = _run_cmd(cmd, os.environ)
            gdb_python_numpy_version = stderr.strip()

    # Display the information
    print(fmt % ("Binary location", config.GDB_BINARY))
    print(fmt % ("Print extension location", print_ext_path))
    print(fmt % ("Python version", gdb_python_version))
    print(fmt % ("NumPy version", gdb_python_numpy_version))
    print_ext_supported = gdb_python_numpy_version is not None
    print(fmt % ("Numba printing extension supported", print_ext_supported))

    print("")
    print("To load the Numba gdb printing extension, execute the following "
          "from the gdb prompt:")
    print(f"\nsource {print_ext_path}\n")
    print('-' * 80)
    warn = """
    =============================================================
    IMPORTANT: Before sharing you should remove any information
    in the above that you wish to keep private e.g. paths.
    =============================================================
    """
    print(dedent(warn))


if __name__ == '__main__':
    display_gdbinfo()
