import sys
import argparse
import os
import subprocess


def get_sys_info():
    # delay these imports until now as they are only needed in this
    # function which then exits.
    import platform
    import json
    import multiprocessing
    from numba.core import config
    from numba import cuda as cu
    from numba.cuda import cudadrv
    from numba.cuda.cudadrv.driver import driver as cudriver
    from numba import roc
    from numba.roc.hlc import hlc, libhlc
    import textwrap as tw
    import ctypes as ct
    import llvmlite.binding as llvmbind
    import locale
    from datetime import datetime
    from itertools import chain
    from subprocess import check_output, CalledProcessError

    try:
        fmt = "%-45s : %-s"
        print("-" * 80)
        print("__Time Stamp__")
        print(datetime.utcnow())
        print("")

        print("__Hardware Information__")
        system_name = platform.system()
        print(fmt % ("Machine", platform.machine()))
        print(fmt % ("CPU Name", llvmbind.get_host_cpu_name()))
        if system_name == 'Linux':
            strmatch = 'Cpus_allowed'
            try:
                loc = '/proc/self/status'
                with open(loc, 'rt') as f:
                    proc_stat = f.read().splitlines()
                    for x in proc_stat:
                        if x.startswith(strmatch):
                            if x.startswith('%s:' % strmatch):
                                hexnum = '0x%s' % x.split(':')[1].strip()
                                acc_cpus = int(hexnum, 16)
                                _n = str(bin(acc_cpus).count('1'))
                                print(fmt % ("Number of accessible CPU cores",
                                                _n))
                            elif x.startswith('%s_list:' % strmatch):
                                _a = x.split(':')[1].strip()
                                print(fmt % ("Listed accessible CPUs cores",
                                                _a))
            except Exception:
                print(fmt % ("CPU count", multiprocessing.cpu_count()))
            # See if CFS is in place
            # https://www.kernel.org/doc/Documentation/scheduler/sched-bwc.txt
            try:
                def scrape_lines(loc):
                    with open(loc, 'rt') as f:
                        return f.read().splitlines()
                loc = '/sys/fs/cgroup/cpuacct/cpu.cfs_period_us'
                cfs_period = int(scrape_lines(loc)[0])
                loc = '/sys/fs/cgroup/cpuacct/cpu.cfs_quota_us'
                cfs_quota = int(scrape_lines(loc)[0])
                if cfs_quota == -1:
                    print(fmt % ("CFS restrictions", "None"))
                else:
                    runtime_amount = float(cfs_quota)/float(cfs_period)
                    print(fmt % ("CFS restrictions (CPUs worth of runtime)",
                                 runtime_amount))
            except Exception:
                print(fmt % ("CFS restrictions", 'Information not available'))
        else:
            print(fmt % ("CPU count", multiprocessing.cpu_count()))

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
        print(fmt % ("System Name", system_name))
        print(fmt % ("Version", platform.version()))
        try:
            if system_name == 'Linux':
                info = ()
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
        lcl = []
        try:
            for x in locale.getdefaultlocale():
                if x is not None:
                    lcl.append(x)
        except Exception as e:
            lcl.append(str(e))
        print(fmt % ("Python Locale ", ' '.join(lcl)))

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
        except Exception as e:
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
        print("__ROC Information__")
        roc_is_available = roc.is_available()
        print(fmt % ("ROC available", roc_is_available))

        toolchains = []
        try:
            libhlc.HLC()
            toolchains.append('librocmlite library')
        except:
            pass
        try:
            cmd = hlc.CmdLine().check_tooling()
            toolchains.append('ROC command line tools')
        except:
            pass

        # if no ROC try and report why
        if not roc_is_available:
            from numba.roc.hsadrv.driver import hsa
            try:
                hsa.is_available
            except Exception as e:
                msg = str(e)
            else:
               msg = 'No ROC toolchains found.'
            print(fmt % ("Error initialising ROC due to", msg))

        if toolchains:
            print(fmt % ("Available Toolchains", ', '.join(toolchains)))

        try:
            # ROC might not be available due to lack of tool chain, but HSA
            # agents may be listed
            from numba.roc.hsadrv.driver import hsa, dgpu_count
            decode = lambda x: x.decode('utf-8') if isinstance(x, bytes) else x
            print("\nFound %s HSA Agents:" % len(hsa.agents))
            for i, agent in enumerate(hsa.agents):
                print('Agent id  : %s' % i)
                print('    vendor: %s' % decode(agent.vendor_name))
                print('      name: %s' % decode(agent.name))
                print('      type: %s' % agent.device)
                print("")

            _dgpus = []
            for a in hsa.agents:
                if a.is_component and a.device == 'GPU':
                   _dgpus.append(decode(a.name))
            print(fmt % ("Found %s discrete GPU(s)" % dgpu_count(), \
                  ', '.join(_dgpus)))
        except Exception as e:
            print("No HSA Agents found, encountered exception when searching:")
            print(e)


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
        print(fmt % ("SVML operational", svml_operational))

        # Check which threading backends are available.
        print("")
        print("__Threading Layer Information__")
        def parse_error(e, backend):
            # parses a linux based error message, this is to provide feedback
            # and hide user paths etc
            try:
                path, problem, symbol =  [x.strip() for x in e.msg.split(':')]
                extn_dso = os.path.split(path)[1]
                if backend in extn_dso:
                    return "%s: %s" % (problem, symbol)
            except Exception:
                pass
            return "Unknown import problem."

        try:
            from numba.np.ufunc import tbbpool
            print(fmt % ("TBB Threading layer available", True))
        except ImportError as e:
            # might be a missing symbol due to e.g. tbb libraries missing
            print(fmt % ("TBB Threading layer available", False))
            print(fmt % ("+--> Disabled due to",
                         parse_error(e, 'tbbpool')))

        try:
            from numba.np.ufunc import omppool
            print(fmt % ("OpenMP Threading layer available", True))
            print(fmt % ("+--> Vendor: ", omppool.openmp_vendor))
        except ImportError as e:
            print(fmt % ("OpenMP Threading layer available", False))
            print(fmt % ("+--> Disabled due to",
                         parse_error(e, 'omppool')))

        try:
            from numba.np.ufunc import workqueue
            print(fmt % ("Workqueue Threading layer available", True))
        except ImportError as e:
            print(fmt % ("Workqueue Threading layer available", False))
            print(fmt % ("+--> Disabled due to",
                         parse_error(e, 'workqueue')))

        # look for numba env vars that are set
        print("")
        print("__Numba Environment Variable Information__")
        _envvar_found = False
        for k, v in os.environ.items():
            if k.startswith('NUMBA_'):
                print(fmt % (k, v))
                _envvar_found = True
        if not _envvar_found:
            print("None set.")

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
