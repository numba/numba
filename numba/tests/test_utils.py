import os
import sys
import threading
from numba.misc import utils
from numba.tests.support import TestCase
from numba import njit
from numba.tests.support import skip_linux


# Solution for capturing stdout
# https://stackoverflow.com/questions/24277488/in-python-how-to-capture-the-stdout-from-a-c-shared-library-to-a-variable
captured_stdout = b''
stdout_pipe = None
def drain_pipe():
	global captured_stdout
	while True:
		data = os.read(stdout_pipe[0], 1024)
		if not data:
			break
		captured_stdout += data


def printf():
	d = i = 392
	utils.printf("d=%d, i=%i\n", d, i)

	u = 7325
	utils.printf("u=%u\n", u)

	o = 610
	utils.printf("o=%o\n", o)

	x = 2042
	utils.printf("x=%x\n", x)
	X = 2042
	utils.printf("X=%X\n", X)

	f = 392.651231
	utils.printf("f=%.3f\n", f)
	F = 392.651231
	utils.printf("F=%.4F\n", F)

	e = 3.9264e+2
	utils.printf("e=%e\n", e)
	E = 3.9264e+2
	utils.printf("E=%E\n", E)

	g = 392.65
	utils.printf("g=%g\n", g)
	G = 392.65
	utils.printf("G=%G\n", G)

	c = 'a'
	s = 'sample'
	utils.printf("c=%s\n", c)
	utils.printf("s=%s\n", s)

	p = 10
	utils.printf("p=%p\n", p)

	utils.fflush()


class TestUtils(TestCase):

	@skip_linux
	def test_printf(self):
		global stdout_pipe
		stdout_fileno = sys.stdout.fileno()
		stdout_save = os.dup(stdout_fileno)
		stdout_pipe = os.pipe()
		os.dup2(stdout_pipe[1], stdout_fileno)
		os.close(stdout_pipe[1])

		d = 3
		cfunc = njit(printf)
		t = threading.Thread(target=drain_pipe)
		t.start()
		cfunc()
		os.close(stdout_fileno)
		t.join()	
		os.close(stdout_pipe[0])
		os.dup2(stdout_save, stdout_fileno)
		os.close(stdout_save)

		expected = 'd=392, i=392\nu=7325\no=1142\nx=7fa\nX=7FA\n' + \
				'f=392.651\nF=392.6512\ne=3.926400e+02\n' + \
				'E=3.926400E+02\ng=392.65\nG=392.65\n' + \
				'c=a\ns=sample\np=0xa\n'

		output = captured_stdout.decode('utf-8')
		self.assertEqual(expected, output)

if __name__ == '__main__':
	unittest.main()