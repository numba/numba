"""
Implements threads using llvm cbuilder. Taken from
numbapro/vectorizers/parallel.py
"""

from llvm.core import *
from llvm_cbuilder import *
import llvm_cbuilder.shortnames as C

import sys

class PThreadAPI(CExternal):
    '''external declaration of pthread API
    '''
    pthread_t = C.void_p

    pthread_create = Type.function(C.int,
                                   [C.pointer(pthread_t),  # thread_t
                                    C.void_p,              # thread attr
                                    C.void_p,              # function
                                    C.void_p])             # arg

    pthread_join = Type.function(C.int, [C.void_p, C.void_p])


class WinThreadAPI(CExternal):
    '''external declaration of pthread API
    '''
    _calling_convention_ = CC_X86_STDCALL

    handle_t = C.void_p

    # lpStartAddress is an LPTHREAD_START_ROUTINE, with the form
    # DWORD ThreadProc (LPVOID lpdwThreadParam )
    CreateThread = Type.function(handle_t,
                                   [C.void_p,            # lpThreadAttributes (NULL for default)
                                    C.intp,              # dwStackSize (0 for default)
                                    C.void_p,            # lpStartAddress
                                    C.void_p,            # lpParameter
                                    C.int32,             # dwCreationFlags (0 for default)
                                    C.pointer(C.int32)]) # lpThreadId (NULL if not required)

    # Return is WAIT_OBJECT_0 (0x00000000) to indicate the thread exited,
    # or WAIT_ABANDONED, WAIT_TIMEOUT, WAIT_FAILED for other conditions.
    WaitForSingleObject = Type.function(C.int32,
                                    [handle_t, # hHandle
                                     C.int32])   # dwMilliseconds (INFINITE == 0xFFFFFFFF means wait forever)

    CloseHandle = Type.function(C.int32, [handle_t])


class ParallelUFuncPosixMixin(object):
    '''ParallelUFunc mixin that implements _dispatch_worker to use pthread.
    '''
    def _dispatch_worker(self, worker, contexts, num_thread):
        api = PThreadAPI(self)
        NULL = self.constant_null(C.void_p)

        threads = self.array(api.pthread_t, num_thread, name='threads')

        # self.debug("launch threads")

        with self.for_range(num_thread) as (loop, i):
            status = api.pthread_create(threads[i].reference(), NULL, worker,
                                        contexts[i].reference().cast(C.void_p))
            with self.ifelse(status != self.constant_null(status.type)) as ifelse:
                with ifelse.then():
                    # self.debug("Error at pthread_create: ", status)
                    self.unreachable()

        with self.for_range(num_thread) as (loop, i):
            status = api.pthread_join(threads[i], NULL)
            with self.ifelse(status != self.constant_null(status.type)) as ifelse:
                with ifelse.then():
                    # self.debug("Error at pthread_join: ", status)
                    self.unreachable()


class ParallelUFuncWindowsMixin(object):
    '''ParallelUFunc mixin that implements _dispatch_worker to use Windows threading.
    '''
    def _dispatch_worker(self, worker, contexts, num_thread):
        api = WinThreadAPI(self)
        NULL = self.constant_null(C.void_p)
        lpdword_NULL = self.constant_null(C.pointer(C.int32))
        zero = self.constant(C.int32, 0)
        intp_zero = self.constant(C.intp, 0)
        INFINITE = self.constant(C.int32, 0xFFFFFFFF)

        threads = self.array(api.handle_t, num_thread, name='threads')

        # self.debug("launch threads")
        # TODO error handling

        with self.for_range(num_thread) as (loop, i):
            threads[i] = api.CreateThread(NULL, intp_zero, worker,
                               contexts[i].reference().cast(C.void_p),
                               zero, lpdword_NULL)

        with self.for_range(num_thread) as (loop, i):
            api.WaitForSingleObject(threads[i], INFINITE)
            api.CloseHandle(threads[i])


if sys.platform == 'win32':
    ParallelMixin = ParallelUFuncWindowsMixin
else:
    ParallelMixin = ParallelUFuncPosixMixin