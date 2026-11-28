import glob
import os
import pickle
import shutil
import subprocess
import sys


CXXFILT = shutil.which('c++filt')


def _print_symbol(label, name):
    indent = ' ' * (len(label) - len(label.lstrip()))
    print('%s%s: %s' % (indent, label.strip(), name))
    if CXXFILT is None or not isinstance(name, str) or '_Z' not in name:
        return
    prefix, symbol = name.split('_Z', 1)
    try:
        result = subprocess.run([CXXFILT, '-n', '_Z' + symbol],
                                capture_output=True, text=True, check=False)
    except OSError:
        return
    demangled = prefix + result.stdout.strip()
    if result.returncode == 0 and demangled != name:
        print('%s  demangled: %s' % (indent, demangled))


def _syms(bitcode):
    try:
        from llvmlite import binding as ll
    except ImportError:
        print('      symbols: unavailable (llvmlite not installed)')
        return
    try:
        mod = ll.parse_bitcode(bitcode)
    except RuntimeError as e:
        raise ValueError('invalid LLVM bitcode') from e
    for fn in mod.functions:
        if not fn.is_declaration:
            _print_symbol('      def', fn.name)


def show_nbi(path):
    print('NBI  %s (%d bytes)' % (path, os.path.getsize(path)))
    with open(path, 'rb') as f:
        version = pickle.load(f)
        stamp, overloads = pickle.loads(f.read())
    if not isinstance(overloads, dict):
        raise ValueError('expected an index mapping')
    print('  version: %s' % version)
    print('  stamp:   %s' % ((stamp.hex() if isinstance(stamp, bytes)
                             else stamp),))
    print('  entries: %d' % len(overloads))
    for index, (key, nbc) in enumerate(overloads.items(), 1):
        if not isinstance(key, tuple) or len(key) not in (3, 4):
            raise ValueError('unexpected index key %r' % (key,))
        print('\n  [%d] %s' % (index, nbc))
        print('      signature: %s' % (key[0],))
        print('      target:    %s' % (key[1],))
        print('      hashes:    %s' % (key[2],))
        if len(key) == 4:
            _print_symbol('      kernel', key[3])


def _show_libdata(libdata):
    if not (isinstance(libdata, tuple) and len(libdata) == 3):
        raise ValueError('expected serialized CodeLibrary')
    name, kind, data = libdata
    print('  library: %s' % name)
    print('  kind:    %s' % kind)
    if kind == 'object' and isinstance(data, tuple) and len(data) == 2:
        obj, bc = data
        print('  object:  %d bytes' % len(obj))
    elif kind == 'bitcode':
        bc = data
    else:
        raise ValueError('unexpected CodeLibrary kind or payload')
    if not isinstance(bc, bytes):
        raise ValueError('expected LLVM bitcode bytes')
    print('  bitcode: %d bytes' % len(bc))
    print('  symbols:')
    _syms(bc)


def show_nbc(path):
    print('NBC  %s (%d bytes)' % (path, os.path.getsize(path)))
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    if not isinstance(obj, tuple):
        raise ValueError('expected a serialized tuple')
    if len(obj) == 3 and obj[1] in ('object', 'bitcode'):
        print('  CodeLibrary\n')
        _show_libdata(obj)
        return
    if len(obj) == 9:
        libdata, fndesc, env, sig = obj[:4]
        print('  CompileResult\n')
        print('  signature: %s' % (sig,))
        name = getattr(fndesc, 'mangled_name', None)
        _print_symbol('  kernel', name)
        llvm_name = getattr(fndesc, 'llvm_func_name', None)
        if llvm_name != name:
            _print_symbol('  llvm_func', llvm_name)
        print('  objectmode: %s\n' % obj[4])
        _show_libdata(libdata)
        return
    raise ValueError('unknown .nbc payload shape')


def main(paths):
    here = os.path.dirname(os.path.abspath(__file__))
    if not paths:
        paths = glob.glob(os.path.join(here, '__pycache__', '*.nb[ic]'))
    if not paths:
        print('No cache files found', file=sys.stderr)
        return 1
    failed = False
    for path in sorted(paths):
        print()
        try:
            if path.endswith('.nbi'):
                show_nbi(path)
            elif path.endswith('.nbc'):
                show_nbc(path)
            else:
                raise ValueError('expected a .nbi or .nbc file')
        except (OSError, EOFError, pickle.UnpicklingError, ValueError) as e:
            print('%s: %s' % (path, e), file=sys.stderr)
            failed = True
    return int(failed)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
