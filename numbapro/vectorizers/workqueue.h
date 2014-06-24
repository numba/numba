typedef struct opaque_thread * thread_pointer;

#ifdef _MSC_VER
    #define DLLEXPORT __declspec(dllexport)
#else
    #define DLLEXPORT
#endif

DLLEXPORT
thread_pointer numba_new_thread(void *worker, void *arg);

DLLEXPORT
int numba_join_thread(thread_pointer thread);


