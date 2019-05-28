"""Expose Numba command via ``python -m numba``."""
import sys
from .numba_entry import main

if __name__ == '__main__':
    sys.exit(main())
