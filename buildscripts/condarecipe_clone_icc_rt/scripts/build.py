import sys
import os
import shutil

libdir = {'w': 'Library',
          'l': 'lib',
          'd': 'lib'}


def run():
    src_dir = os.environ.get('SRC_DIR')
    prefix = os.environ.get('PREFIX')

    libd = libdir.get(sys.platform[0], None)
    assert libd is not None

    # remove 'lib' from the prefix so a direct copy from the original
    # package can be made
    lib_dir = os.path.join(prefix, libd)
    shutil.rmtree(lib_dir)
    # copy in the original package lib dir
    shutil.copytree(os.path.join(src_dir, libd), lib_dir)

    # and copy the license
    info_dir = os.path.join(src_dir, 'info')
    shutil.copy(os.path.join(info_dir, 'LICENSE.txt'), src_dir)
    shutil.rmtree(info_dir)


if __name__ == "__main__":
    args = sys.argv
    assert len(args) == 1
    run()
