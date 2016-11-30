import os
from os.path import dirname, join, abspath
from unittest.case import TestCase
from unittest.suite import TestSuite
from subprocess import STDOUT, check_output, CalledProcessError

from numba.testing.ddt import ddt, data
from numba.testing.notebook import NotebookTest
from numba import cuda

# setup coverage
default_config_file = abspath(join(dirname(dirname(__file__)), '.coveragerc'))
print('using coveragerc:', default_config_file)
os.environ['COVERAGE_PROCESS_START'] = default_config_file


test_scripts = [
    'binarytree.py',
    'bubblesort.py',
    'cffi_example.py',
    'compile_with_pycc.py',
    'ctypes_example.py',
    'fbcorr.py',
    'jitclass.py',
    'linkedlist.py',
    'movemean.py',
    'nogil.py',
    'objects.py',
    'ra24.py',
    'stack.py',
    'structures.py',
    'sum.py',
    'ufuncs.py',
    'blackscholes/blackscholes.py',
    'blackscholes/blackscholes_numba.py',
    'laplace2d/laplace2d.py',
    'laplace2d/laplace2d-numba.py',
    'blur_image.py',
    'mergesort.py',
    'mandel/mandel_vectorize.py',
    'mandel/mandel_jit.py',
    'nbody/nbody.py',
    'nbody/nbody_modified_by_MarkHarris.py',
    'vectorize/sum.py',
    'vectorize/polynomial.py',
]

if cuda.is_available():
    test_scripts.extend([
    'blackscholes/blackscholes_cuda.py',
    'cudajit/matmul.py',
    'cudajit/matmul_smem.py',
    'cudajit/sum.py',
    'laplace2d/laplace2d-numba-cuda.py',
    'laplace2d/laplace2d-numba-cuda-improve.py',
    'laplace2d/laplace2d-numba-cuda-smem.py',
    'vectorize/cuda_polynomial.py',
    # 'cuda_mpi.py',
    ])

notebooks = ['j0 in Numba.ipynb',
             'LinearRegr.ipynb',
             'numba.ipynb',
             'Using Numba.ipynb']


@ddt
class TestExample(TestCase):
    """Test adapter to validate example applets."""

    def setUp(self):
        # to pick up sitecustomize.py
        basedir = dirname(__file__)
        os.environ['PYTHONPATH'] = basedir
        # matplotlibrc to suppress display
        os.environ['MATPLOTLIBRC'] = basedir

    @data(*test_scripts)
    def test(self, script):
        script = abspath(join(dirname(dirname(__file__)), script))
        status = 0
        try:
            print(script)
            out = check_output(script, stderr=STDOUT, shell=True)
        except CalledProcessError as e:
            status = e.returncode
            out = e.output

        print(out.decode())
        self.assertEqual(status, 0)


@ddt
class NBTest(NotebookTest):

    @data(*notebooks)
    def test(self, nb):
        test = 'check_error'  # This is the only currently supported test type
        notebook = join(dirname(dirname(__file__)), 'notebooks', nb)
        self._test_notebook(notebook, test)


def load_tests(loader, tests, pattern):
    notebooks = loader.loadTestsFromTestCase(NBTest)
    examples = loader.loadTestsFromTestCase(TestExample)
    return TestSuite([notebooks, examples])

