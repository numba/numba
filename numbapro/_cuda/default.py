# When imported this module either creates a default cuda context,
# or it does nothing if there is already one.
# Other modules may import this module to ensure the existance
# of a CUDA context.

from .driver import Driver, Device
import logging, os
logger = logging.getLogger(__name__)

# Let there be a CUDA driver
# Otherwise, raise ImportError
driver = Driver()

device_number = 0  # default device id

## Choose the first device unless the user has a specific choice
## stored in file ~/.cuda_device
#
#try:
#    n = open(os.path.expanduser("~/.cuda_device")).read().strip()
#except IOError:
#    pass
#else:
#    try:
#        device_number = int(n)
#    except ValueError:
#        logger.error("Failed to parse ~/.cuda_device: %s" % n)

if driver.current_context(noraise=True) is None:
    device = Device(device_number)
    driver.create_context(device)


