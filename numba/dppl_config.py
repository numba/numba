try:
    import dpctl
    dppl_present = dpctl.has_gpu_queues() and dpctl.has_cpu_queues()
except:
    dppl_present = False
