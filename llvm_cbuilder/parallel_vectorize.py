from llvm.core import *
from llvm.passes import *

from llvm_cbuilder import *
import llvm_cbuilder.shortnames as C

class WorkQueue(CStruct):
    _fields_ = [
        ('next', C.intp),
        ('last', C.intp),
        ('lock', C.int),
    ]


    def Lock(self):
        with self.parent.loop() as loop:
            with loop.condition() as setcond:
                unlocked = self.parent.constant(self.lock.type, 0)
                locked = self.parent.constant(self.lock.type, 1)

                res = self.lock.reference().atomic_cmpxchg(unlocked, locked,
                                               ordering='acquire')
                setcond( res != unlocked )

            with loop.body():
                pass

    def Unlock(self):
        unlocked = self.parent.constant(self.lock.type, 0)
        locked = self.parent.constant(self.lock.type, 1)

        res = self.lock.reference().atomic_cmpxchg(locked, unlocked,
                                                   ordering='release')

        with self.parent.ifelse( res != locked ) as ifelse:
            with ifelse.then():
                # This shall kill the program
                self.parent.unreachable()


class ContextCommon(CStruct):
    _fields_ = [
        # loop ufunc args
        ('args',        C.pointer(C.char_p)),
        ('dimensions',  C.pointer(C.intp)),
        ('steps',       C.pointer(C.intp)),
        ('data',        C.void_p),
        # specifics for work queues
        ('func',        C.void_p),
        ('num_thread',  C.int),
        ('workqueues',  C.pointer(WorkQueue.llvm_type())),
    ]

class Context(CStruct):
    _fields_ = [
        ('common',    C.pointer(ContextCommon.llvm_type())),
        ('id',        C.int),
        ('completed', C.intp),
    ]

class ParallelUFunc(CDefinition):
    _name_   = 'parallel_ufunc'
    _retty_  = C.void
    _argtys_ = [
        ('func',       C.void_p),
        ('worker',     C.void_p),
        ('args',       C.pointer(C.char_p)),
        ('dimensions', C.pointer(C.intp)),
        ('steps',      C.pointer(C.intp)),
        ('data',       C.void_p),
    ]

    def body(self, func, worker, args, dimensions, steps, data, ThreadCount=1):


        common = self.var(ContextCommon, name='common')
        workqueues = self.array(WorkQueue, ThreadCount, name='workqueues')
        contexts = self.array(Context, ThreadCount, name='contexts')

        num_thread = self.var(C.int, ThreadCount, name='num_thread')

        common.args.assign(args)
        common.dimensions.assign(dimensions)
        common.steps.assign(steps)
        common.data.assign(data)
        common.func.assign(func)
        common.num_thread.assign(num_thread.cast(C.int))
        common.workqueues.assign(workqueues.reference())

        N = dimensions[0]
        ChunkSize = self.var_copy(N / num_thread.cast(N.type))
        ChunkSize_NULL = self.constant_null(ChunkSize.type)
        with self.ifelse(ChunkSize == ChunkSize_NULL) as ifelse:
            with ifelse.then():
                ChunkSize.assign(self.constant(ChunkSize.type, 1))
                num_thread.assign(N.cast(num_thread.type))

        self._populate_workqueues(workqueues, N, ChunkSize, num_thread)
        self._populate_context(contexts, common, num_thread)

        self._dispatch_worker(worker, contexts,  num_thread)

        # Check for race condition
        total_completed = self.var(C.intp, 0, name='total_completed')
        for t in range(ThreadCount):
            cur_ctxt = contexts[t].as_struct(Context)
            total_completed += cur_ctxt.completed
            self.debug(cur_ctxt.id, 'completed', cur_ctxt.completed)

        with self.ifelse( total_completed == N ) as ifelse:
            with ifelse.then():
                self.debug("All is well!")
            with ifelse.otherwise():
                self.debug("ERROR: race occurred!")

        self.ret()

    def _populate_workqueues(self, workqueues, N, ChunkSize, num_thread):
        i = self.var(num_thread.type, 0, name='i')
        ONE = self.constant(i.type, 1)
        with self.loop() as loop:
            with loop.condition() as setcond:
                setcond( i < num_thread )
            with loop.body():
                cur_wq = workqueues[i].as_struct(WorkQueue)
                cur_wq.next.assign(i.cast(ChunkSize.type) * ChunkSize)
                cur_wq.last.assign((i + ONE).cast(ChunkSize.type) * ChunkSize)
                cur_wq.lock.assign(self.constant(C.int, 0))
                # increment
                i += ONE
        # end loop
        last_wq = workqueues[num_thread - ONE].as_struct(WorkQueue)
        last_wq.last.assign(N)

    def _populate_context(self, contexts, common, num_thread):
        i = self.var(num_thread.type, 0, name='i')
        ONE = self.constant(i.type, 1)
        with self.loop() as loop:
            with loop.condition() as setcond:
                setcond( i < num_thread )
            with loop.body():
                cur_ctxt = contexts[i].as_struct(Context)
                cur_ctxt.common.assign(common.reference())
                cur_ctxt.id.assign(i)
                cur_ctxt.completed.assign(self.constant_null(
                                                    cur_ctxt.completed.type))
                # increment
                i += ONE


class ParallelUFuncPosix(ParallelUFunc):
    def _dispatch_worker(self, worker, contexts, num_thread):
        api = PThreadAPI(self)
        NULL = self.constant_null(C.void_p)

        threads = self.array(api.pthread_t, num_thread, name='threads')

        # self.debug("launch threads")
        # TODO error handling

        i = self.var(num_thread.type, 0, name='i')
        ONE = self.constant(i.type, 1)
        with self.loop() as loop:
            with loop.condition() as setcond:
                setcond( i < num_thread )
            with loop.body():
                api.pthread_create(threads[i].reference(), NULL, worker,
                                   contexts[i].reference().cast(C.void_p))
                i += ONE

        # self.debug("join threads")
        i.assign(self.constant_null(i.type))
        with self.loop() as loop:
            with loop.condition() as setcond:
                setcond( i < num_thread )
            with loop.body():
                api.pthread_join(threads[i], NULL)
                i += ONE
        # self.debug("closing")

class UFuncCore(CDefinition):
    _name_ = 'ufunc_worker'
    _argtys_ = [
        ('context', C.pointer(Context.llvm_type())),
        ]

    def body(self, context):
        context = context.as_struct(Context)
        common = context.common.as_struct(ContextCommon)
        tid = context.id

        # self.debug("start thread", tid, "/", common.num_thread)
        workqueue = common.workqueues[tid].as_struct(WorkQueue)

        self._do_workqueue(common, workqueue, tid, context.completed)
        self._do_work_stealing(common, tid, context.completed) # optional

        self.ret()

    def _do_workqueue(self, common, workqueue, tid, completed):
        '''
        Process local workqueue.
        '''
        ZERO = self.constant_null(C.int)

        with self.loop() as loop:
            with loop.condition() as setcond:
                setcond(ZERO == ZERO)  # loop forever

            with loop.body():
                workqueue.Lock()
                # Critical section
                item = self.var_copy(workqueue.next, name='item')
                workqueue.next += self.constant(item.type, 1)
                last = self.var_copy(workqueue.last, name='last')
                # Release
                workqueue.Unlock()

                with self.ifelse( item >= last ) as ifelse:
                    with ifelse.then():
                        loop.break_loop()

                self._do_work(common, item, tid, completed)

    def _do_work_stealing(self, common, tid, completed):
        '''
        Steal work from other workqueues
        '''
        # self.debug("start work stealing", tid)
        incomplete_thread_ct = self.var(C.int, 1)
        ITC_NULL = self.constant_null(incomplete_thread_ct.type)
        with self.loop() as loop:
            with loop.condition() as setcond:
                setcond( incomplete_thread_ct != ITC_NULL )

            with loop.body():
                incomplete_thread_ct.assign(ITC_NULL)
                self._do_work_stealing_innerloop(common, incomplete_thread_ct,
                                                 tid, completed)

    def _do_work_stealing_innerloop(self, common, incomplete_thread_ct, tid,
                                    completed):
        i = self.var(C.int, 0)
        with self.loop() as loop:
            with loop.condition() as setcond:
                setcond( i < common.num_thread )

            with loop.body():
                with self.ifelse( i != tid ) as ifelse:
                    with ifelse.then():
                        otherqueue = common.workqueues[i].as_struct(WorkQueue)
                        self._do_work_stealing_check(common, otherqueue,
                                                     incomplete_thread_ct, tid,
                                                     completed)

                # increment
                i += self.constant(i.type, 1)

    def _do_work_stealing_check(self, common, otherqueue,
                                incomplete_thread_ct, tid, completed):
        otherqueue.Lock()
        ONE = self.constant(otherqueue.last.type, 1)
        ITC_ONE = self.constant(incomplete_thread_ct.type, 1)
        with self.ifelse(otherqueue.next < otherqueue.last) as ifelse:
            with ifelse.then():
                otherqueue.last -= ONE
                item = self.var_copy(otherqueue.last)

                otherqueue.Unlock()
                # Released

                self._do_work(common, item, tid, completed)

                # Mark incomplete thread
                incomplete_thread_ct.assign(ITC_ONE)

            with ifelse.otherwise():
                otherqueue.Unlock()

    def _do_work(self, common, item, tid, completed):
        # self.debug("   tid", tid, "item", item)
        completed += self.constant(completed.type, 1)


class PThreadAPI(CExternal):
    pthread_t = C.void_p

    pthread_create = Type.function(C.int,
                                   [C.pointer(pthread_t),  # thread_t
                                    C.void_p,              # thread attr
                                    C.void_p,              # function
                                    C.void_p])             # arg

    pthread_join = Type.function(C.int, [C.void_p, C.void_p])


def _main():
    NUM_THREAD = 2
    module = Module.new(__name__)

    fpm = FunctionPassManager.new(module)
    PassManagerBuilder.new().populate(fpm)

    f1 = ParallelUFuncPosix.define(module, ThreadCount=NUM_THREAD)
    f2 = UFuncCore.define(module)
    #print(module)
#    print(f1)
    print(f2)
    module.verify()

    fpm.run(f1)
    fpm.run(f2)
    print('optimized'.center(80,'-'))
    print(f2)

if __name__ == '__main__':
    _main()

