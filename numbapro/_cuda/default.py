from .driver import Driver, Device
import logging, os
logger = logging.getLogger(__name__)

# Let there be a CUDA driver
# Otherwise, raise ImportError
driver = Driver()

# Choose the first device unless the user has a specific choice
# stored in file ~/.cuda_device
device_number = 0
try:
    n = open(os.path.expanduser("~/.cuda_device")).read().strip()
except IOError:
    pass
else:
    try:
        device_number = int(n)
    except ValueError:
        logger.error("Failed to parse ~/.cuda_device: %s" % n)

device = Device(device_number)
driver.create_context(device)


