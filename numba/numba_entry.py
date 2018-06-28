from __future__ import print_function, division, absolute_import

import sys
import argparse
import os
import subprocess


def get_sys_info():
    # delay these imports until now as they are only needed in this
    # function which then exits.
    import platform
    import json
    from numba import config
    from numba import cuda as cu
    from numba.cuda import cudadrv
    from numba.cuda.cudadrv.driver import driver as cudriver
    import textwrap as tw
    import ctypes as ct
    import llvmlite.binding as llvmbind
    import locale
    from datetime import datetime
    from itertools import chain
    from subprocess import check_output, CalledProcessError

    try:
        fmt = "%-35s : %-s"
        print("-" * 80)
        print("__Time Stamp__")
        print(datetime.utcnow())
        print("")

        print("__Hardware Information__")
        print(fmt % ("Machine", platform.machine()))
        print(fmt % ("CPU Name", llvmbind.get_host_cpu_name()))
        try:
            featuremap = llvmbind.get_host_cpu_features()
        except RuntimeError:
            print(fmt % ("CPU Features", "NA"))
        else:
            features = sorted([key for key, value in featuremap.items()
                               if value])
            cpu_feat = tw.fill(' '.join(features), 80)
            print(fmt % ("CPU Features", ""))
            print(cpu_feat)
        print("")

        print("__OS Information__")
        print(fmt % ("Platform", platform.platform(aliased=True)))
        print(fmt % ("Release", platform.release()))
        system_name = platform.system()
        print(fmt % ("System Name", system_name))
        print(fmt % ("Version", platform.version()))
        try:
            if system_name == 'Linux':
                info = platform.linux_distribution()
            elif system_name == 'Windows':
                info = platform.win32_ver()
            elif system_name == 'Darwin':
                info = platform.mac_ver()
            else:
                raise RuntimeError("Unknown system.")
            buf = ''.join([x
                           if x != '' else ' '
                           for x in list(chain.from_iterable(info))])
            print(fmt % ("OS specific info", buf))

            if system_name == 'Linux':
                print(fmt % ("glibc info", ' '.join(platform.libc_ver())))
        except:
            print("Error: System name incorrectly identified or unknown.")
        print("")

        print("__Python Information__")
        print(fmt % ("Python Compiler", platform.python_compiler()))
        print(
            fmt %
            ("Python Implementation",
             platform.python_implementation()))
        print(fmt % ("Python Version", platform.python_version()))
        print(
            fmt %
            ("Python Locale ", ' '.join(
                [x for x in locale.getdefaultlocale() if x is not None])))

        print("")
        print("__LLVM information__")
        print(
            fmt %
            ("LLVM version", '.'.join(
                [str(k) for k in llvmbind.llvm_version_info])))

        print("")
        print("__CUDA Information__")
        # Look for GPUs
        try:
            cu.list_devices()[0]  # will a device initialise?
        except BaseException as e:
            msg_not_found = "CUDA driver library cannot be found"
            msg_disabled_by_user = "CUDA is disabled"
            msg_end = " or no CUDA enabled devices are present."
            msg_generic_problem = "Error: CUDA device intialisation problem."
            msg = getattr(e, 'msg', None)
            if msg is not None:
                if msg_not_found in msg:
                    err_msg = msg_not_found + msg_end
                elif msg_disabled_by_user in msg:
                    err_msg = msg_disabled_by_user + msg_end
                else:
                    err_msg = msg_generic_problem + " Message:" + msg
            else:
                err_msg = msg_generic_problem + " " + str(e)
            # Best effort error report
            print("%s\nError class: %s" % (err_msg, str(type(e))))
        else:
            try:
                cu.detect()
                dv = ct.c_int(0)
                cudriver.cuDriverGetVersion(ct.byref(dv))
                print(fmt % ("CUDA driver version", dv.value))
                print("CUDA libraries:")
                cudadrv.libs.test(sys.platform, print_paths=False)
            except:
                print(
                    "Error: Probing CUDA failed (device and driver present, runtime problem?)\n")

        print("")
        print("__SVML Information__")
        # replicate some SVML detection logic from numba.__init__ here.
        # if SVML load fails in numba.__init__ the splitting of the logic
        # here will help diagnosis of the underlying issue
        have_svml_library = True
        try:
            if sys.platform.startswith('linux'):
                llvmbind.load_library_permanently("libsvml.so")
            elif sys.platform.startswith('darwin'):
                llvmbind.load_library_permanently("libsvml.dylib")
            elif sys.platform.startswith('win'):
                llvmbind.load_library_permanently("svml_dispmd")
            else:
                have_svml_library = False
        except:
            have_svml_library = False
        func = getattr(llvmbind.targets, "has_svml", None)
        llvm_svml_patched = func() if func is not None else False
        svml_operational = (config.USING_SVML and llvm_svml_patched \
                            and have_svml_library)
        print(fmt % ("SVML state, config.USING_SVML", config.USING_SVML))
        print(fmt % ("SVML library found and loaded", have_svml_library))
        print(fmt % ("llvmlite using SVML patched LLVM", llvm_svml_patched))
        print(fmt % ("SVML operational:", svml_operational))

        # Look for conda and conda information
        print("")
        print("__Conda Information__")
        cmd = ["conda", "info", "--json"]
        try:
            conda_out = check_output(cmd)
        except Exception as e:
            print(
                "Conda not present/not working.\nError was %s\n" % e)
        else:
            data = ''.join(conda_out.decode("utf-8").splitlines())
            jsond = json.loads(data)
            keys = ['conda_build_version',
                    'conda_env_version',
                    'platform',
                    'python_version',
                    'root_writable']
            for k in keys:
                try:
                    print(fmt % (k, jsond[k]))
                except KeyError:
                    pass

            # get info about current environment
            cmd = ["conda", "list"]
            try:
                conda_out = check_output(cmd)
            except CalledProcessError as e:
                print("Error: Conda command failed. Error was %s\n" % e.output)
            else:
                print("")
                print("__Current Conda Env__")
                data = conda_out.decode("utf-8").splitlines()
                for k in data:
                    if k[0] != '#':  # don't show where the env is, personal data
                        print(k)

        print("-" * 80)

    except Exception as e:
        print("Error: The system reporting tool has failed unexpectedly.")
        print("Exception was:")
        print(e)

    finally:
        print(
            "%s" %
            "If requested, please copy and paste the information between\n"
            "the dashed (----) lines, or from a given specific section as\n"
            "appropriate.\n\n"
            "=============================================================\n"
            "IMPORTANT: Please ensure that you are happy with sharing the\n"
            "contents of the information present, any information that you\n"
            "wish to keep private you should remove before sharing.\n"
            "=============================================================\n")


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotate', help='Annotate source',
                        action='store_true')
    parser.add_argument('--dump-llvm', action="store_true",
                        help='Print generated llvm assembly')
    parser.add_argument('--dump-optimized', action='store_true',
                        help='Dump the optimized llvm assembly')
    parser.add_argument('--dump-assembly', action='store_true',
                        help='Dump the LLVM generated assembly')
    parser.add_argument('--dump-cfg', action="store_true",
                        help='[Deprecated] Dump the control flow graph')
    parser.add_argument('--dump-ast', action="store_true",
                        help='[Deprecated] Dump the AST')
    parser.add_argument('--annotate-html', nargs=1,
                        help='Output source annotation as html')
    parser.add_argument('-s', '--sysinfo', action="store_true",
                        help='Output system information for bug reporting')
    parser.add_argument('filename', nargs='?', help='Python source filename')
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()

    if args.dump_cfg:
        print("CFG dump is removed.")
        sys.exit(1)
    if args.dump_ast:
        print("AST dump is removed.  Numba no longer depends on AST.")
        sys.exit(1)

    if args.sysinfo:
        print("System info:")
        get_sys_info()
        sys.exit(0)

    os.environ['NUMBA_DUMP_ANNOTATION'] = str(int(args.annotate))
    if args.annotate_html is not None:
        try:
            from jinja2 import Template
        except ImportError:
            raise ImportError("Please install the 'jinja2' package")
        os.environ['NUMBA_DUMP_HTML'] = str(args.annotate_html[0])
    os.environ['NUMBA_DUMP_LLVM'] = str(int(args.dump_llvm))
    os.environ['NUMBA_DUMP_OPTIMIZED'] = str(int(args.dump_optimized))
    os.environ['NUMBA_DUMP_ASSEMBLY'] = str(int(args.dump_assembly))

    if args.filename:
        cmd = [sys.executable, args.filename]
        subprocess.call(cmd)
    else:
        print("numba: error: the following arguments are required: filename")
        sys.exit(1)
