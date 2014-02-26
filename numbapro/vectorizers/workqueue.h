#ifdef _MSC_VER
    #define DLLEXPORT __declspec(dllexport)
#else
    #define DLLEXPORT
#endif

DLLEXPORT
void* numba_new_thread(void *worker, void *arg);

DLLEXPORT
int numba_join_thread(void *thread);

