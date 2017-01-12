from __future__ import print_function

from contextlib import contextmanager
import sys
import threading

import numpy as np

from numba.six import reraise
from .array import to_device
from .kernelapi import Dim3, FakeCUDAModule, swapped_cuda_module
from ..errors import normalize_kernel_dimensions


"""
Global variable to keep track of the current "kernel context", i.e the
FakeCUDAModule.  We only support one kernel launch at a time.
No support for concurrent kernel launch.
"""
_kernel_context = None


@contextmanager
def _push_kernel_context(mod):
    """
    Push the current kernel context.
    """
    global _kernel_context
    assert _kernel_context is None, "conrrent simulated kernel not supported"
    _kernel_context = mod
    try:
        yield
    finally:
        _kernel_context = None


def _get_kernel_context():
    """
    Get the current kernel context. This is usually done by a device function.
    """
    return _kernel_context


class FakeCUDAKernel(object):
    '''
    Wraps a @cuda.jit-ed function.
    '''

    def __init__(self, fn, device, fastmath=False):
        self.fn = fn
        self._device = device
        self._fastmath = fastmath
        # Initial configuration: 1 block, 1 thread, stream 0, no dynamic shared
        # memory.
        self[1, 1, 0, 0]

    def __call__(self, *args):
        if self._device:
            with swapped_cuda_module(self.fn, _get_kernel_context()):
                return self.fn(*args)

        fake_cuda_module = FakeCUDAModule(self.grid_dim, self.block_dim,
                                          self.dynshared_size)
        with _push_kernel_context(fake_cuda_module):
            # fake_args substitutes all numpy arrays for FakeCUDAArrays
            # because they implement some semantics differently
            def fake_arg(arg):
                if isinstance(arg, np.ndarray) and arg.ndim > 0:
                    return to_device(arg)
                return arg
            fake_args = [fake_arg(arg) for arg in args]

            with swapped_cuda_module(self.fn, fake_cuda_module):
                # Execute one block at a time
                for grid_point in np.ndindex(*self.grid_dim):
                    bm = BlockManager(self.fn, self.grid_dim, self.block_dim)
                    bm.run(grid_point, *fake_args)

    def __getitem__(self, configuration):
        self.grid_dim, self.block_dim = \
            normalize_kernel_dimensions(*configuration[:2])

        if len(configuration) == 4:
            self.dynshared_size = configuration[3]

        return self

    def bind(self):
        pass

    @property
    def ptx(self):
        '''
        Required in order to proceed through some tests, but serves no functional
        purpose.
        '''
        res = '.const'
        res += '\n.local'
        if self._fastmath:
            res += '\ndiv.full.ftz.f32'
        return res




# Thread emulation


class BlockThread(threading.Thread):
    '''
    Manages the execution of a function for a single CUDA thread.
    '''
    def __init__(self, f, manager, blockIdx, threadIdx):
        super(BlockThread, self).__init__(target=f)
        self.syncthreads_event = threading.Event()
        self.syncthreads_blocked = False
        self._manager = manager
        self.blockIdx = Dim3(*blockIdx)
        self.threadIdx = Dim3(*threadIdx)
        self.exception = None

    def run(self):
        try:
            super(BlockThread, self).run()
        except Exception as e:
            tid = 'tid=%s' % list(self.threadIdx)
            ctaid = 'ctaid=%s' % list(self.blockIdx)
            if str(e) == '':
                msg = '%s %s' % (tid, ctaid)
            else:
                msg = '%s %s: %s' % (tid, ctaid, e)
            tb = sys.exc_info()[2]
            self.exception = (type(e), type(e)(msg), tb)

    def syncthreads(self):
        self.syncthreads_blocked = True
        self.syncthreads_event.wait()
        self.syncthreads_event.clear()

    def __str__(self):
        return 'Thread <<<%s, %s>>>' % (self.blockIdx, self.threadIdx)


class BlockManager(object):
    '''
    Manages the execution of a thread block.

    When run() is called, all threads are started. Each thread executes until it
    hits syncthreads(), at which point it sets its own syncthreads_blocked to
    True so that the BlockManager knows it is blocked. It then waits on its
    syncthreads_event.

    The BlockManager polls threads to determine if they are blocked in
    syncthreads(). If it finds a blocked thread, it adds it to the set of
    blocked threads. When all threads are blocked, it unblocks all the threads.
    The thread are unblocked by setting their syncthreads_blocked back to False
    and setting their syncthreads_event.

    The polling continues until no threads are alive, when execution is
    complete.
    '''
    def __init__(self, f, grid_dim, block_dim):
        self._grid_dim = grid_dim
        self._block_dim = block_dim
        self._f = f

    def run(self, grid_point, *args):
        # Create all threads
        threads = set()
        livethreads = set()
        blockedthreads = set()
        for block_point in np.ndindex(*self._block_dim):
            def target():
                self._f(*args)
            t = BlockThread(target, self, grid_point, block_point)
            t.start()
            threads.add(t)
            livethreads.add(t)

        # Potential optimisations:
        # 1. Continue the while loop immediately after finding a blocked thread
        # 2. Don't poll already-blocked threads
        while livethreads:
            for t in livethreads:
                if t.syncthreads_blocked:
                    blockedthreads.add(t)
                elif t.exception:
                    reraise(*(t.exception))
            if livethreads == blockedthreads:
                for t in blockedthreads:
                    t.syncthreads_blocked = False
                    t.syncthreads_event.set()
                blockedthreads = set()
            livethreads = set([ t for t in livethreads if t.is_alive() ])
        # Final check for exceptions in case any were set prior to thread
        # finishing, before we could check it
        for t in threads:
            if t.exception:
                reraise(*(t.exception))
