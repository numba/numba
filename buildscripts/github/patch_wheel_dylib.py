import sys
import re
import subprocess as subp


known_rpaths = {
    '@rpath/libc++.1.dylib': "/usr/lib/libc++.1.dylib",
}


def run_shell(cmd):
    out = subp.check_output(cmd, shell=True)
    return out


def main(whl):
    # Find shared libraries
    run_shell('wheel unpack {}'.format(whl))
    thedir = '-'.join(whl.split('-')[:2])

    sharedlibs = run_shell(
        'find {} -name "*.so"'.format(thedir)).decode().splitlines()
    # Scan paths
    regex_rpath = re.compile(r'@rpath\/[^ ]+')
    updated_files = set()
    for path in sharedlibs:
        # Check rpaths
        otool_out = run_shell('otool -L {}'.format(path)).decode()
        rpaths = set(regex_rpath.findall(otool_out))
        unknown = rpaths - set(known_rpaths.keys())
        if unknown:
            #raise AssertionError('unknown rpath {}'.format(unknown))
            print('unknown', unknown)
        # Patch
        for rpath in rpaths & set(known_rpaths.keys()):
            print('patch', rpath, 'in', path)
            run_shell('install_name_tool -change {} {} {}'.format(
                rpath, known_rpaths[rpath], path))
            updated_files.add(path)

        # Remove absolute LC_RPATH entries left over from the build environment.
        # otool -l emits LC_RPATH as exactly 3 consecutive lines:
        #   cmd LC_RPATH
        #   cmdsize 56
        #   path /Users/runner/miniconda3/envs/test/lib (offset 12)
        # so path is always at index i+2 relative to the cmd line.
        lc_lines = run_shell('otool -l {}'.format(path)).decode().splitlines()
        for i, line in enumerate(lc_lines):
            if 'cmd LC_RPATH' in line:
                lc_rpath = lc_lines[i + 2].split()[1]
                if lc_rpath.startswith('/'):
                    print('delete LC_RPATH', lc_rpath, 'from', path)
                    run_shell('install_name_tool -delete_rpath {} {}'.format(
                        lc_rpath, path))
                    updated_files.add(path)

    print('updated_files', updated_files)
    # Update whl
    if updated_files:
        run_shell('wheel pack {}'.format(thedir))
    else:
        print('nothing to do')


if __name__ == '__main__':
    main(*sys.argv[1:])
