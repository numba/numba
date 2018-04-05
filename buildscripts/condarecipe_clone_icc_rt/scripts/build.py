import subprocess
import sys
import os
import shutil
import tarfile
import tempfile


def run():
    version = os.environ.get('PKG_VERSION')
    build_no = os.environ.get('PKG_BUILDNUM')
    src_dir = os.environ.get('SRC_DIR')
    sys_prefix = os.environ.get('SYS_PREFIX')
    prefix = os.environ.get('PREFIX')
    conda = os.path.join(sys_prefix, "bin", "conda")

    pkg_spec = "icc_rt=%s=intel_%s" % (version,  build_no)

    # force cache update, this shouldn't impact the PREFIX dir
    cmd = [conda, "create", "-c", "intel", "-y", "-p", "junk", "--no-deps",
           "--download-only", "python", pkg_spec]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE)

    # grab package from cache
    tarbz_name = pkg_spec.replace('=', '-') + ".tar.bz2"
    tar_file = os.path.join(sys_prefix, "pkgs", tarbz_name)

    # extract it to a tmpdir
    with tempfile.TemporaryDirectory() as tar_tmpd:
        with tarfile.open(tar_file, "r:bz2") as tar:
            tar.extractall(tar_tmpd)
        # remove 'lib' from the prefix so a direct copy from the original
        # package can be made
        lib_dir = os.path.join(prefix, 'lib')
        shutil.rmtree(lib_dir)
        # copy in the original package lib dir
        shutil.copytree(os.path.join(tar_tmpd, 'lib'), lib_dir)
        # and copy the license
        shutil.copy(os.path.join(tar_tmpd, 'info', 'LICENSE.txt'), src_dir)


if __name__ == "__main__":
    args = sys.argv
    assert len(args) == 1
    run()
