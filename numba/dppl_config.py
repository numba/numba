try:
    import dpctl
    dppl_present = dpctl.has_sycl_platforms() and dpctl.has_gpu_queues()
except:
    dppl_present = False
