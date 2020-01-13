#include <numba_oneapi_glue.h>
#include <stdio.h>

int main (int argc, char** argv)
{
    runtime_t rt;
    int status;

    status = create_numba_oneapi_runtime(&rt);

    if(status == NUMBA_ONEAPI_FAILURE)
        goto error;

    dump_numba_oneapi_runtime_info(rt);
    printf("Printing First CPU device info: \n");
    dump_device_info(&rt->first_cpu_device);
    printf("Printing First GPU device info: \n");
    dump_device_info(&rt->first_gpu_device);
    destroy_numba_oneapi_runtime(&rt);

    return 0;

error:
    return NUMBA_ONEAPI_FAILURE;
}
