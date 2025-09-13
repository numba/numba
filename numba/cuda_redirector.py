import importlib
import importlib.abc
import pathlib
import sys
import warnings

multiple_locations_msg = (
    "Multiple submodule search locations for {}. "
    "Cannot redirect numba.cuda to numba_cuda"
)

no_spec_msg = (
    "The NVIDIA-maintained CUDA target (the `numba_cuda` module) has not been "
    "found. Falling back to the built-in CUDA target. The NVIDIA-maintained "
    "target should be installed - see https://numba.readthedocs.io/en/stable/cuda/overview.html#cuda-deprecation-status"
    " - for installation instructions, see https://nvidia.github.io/numba-cuda/user/installation.html"
)


class NumbaCudaFinder(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.initialized = None

    def ensure_initialized(self):
        if self.initialized is not None:
            return self.initialized

        numba_spec = importlib.util.find_spec("numba")
        numba_cuda_spec = importlib.util.find_spec("numba_cuda")

        if numba_cuda_spec is None:
            warnings.warn(no_spec_msg, DeprecationWarning)
            self.initialized = False
            return False

        numba_search_locations = numba_spec.submodule_search_locations
        numba_cuda_search_locations = numba_cuda_spec.submodule_search_locations

        if len(numba_search_locations) != 1:
            warnings.warn(multiple_locations_msg.format("numba"))
            self.initialized = False
            return False

        if len(numba_cuda_search_locations) != 1:
            warnings.warn(multiple_locations_msg.format("numba_cuda"))
            self.initialized = False
            return False

        self.numba_path = numba_search_locations[0]

        location = numba_cuda_search_locations[0]
        self.numba_cuda_path = str((pathlib.Path(location) / "numba"))

        self.initialized = True
        return True

    def find_spec(self, name, path, target=None):
        if "numba.cuda" in name:
            initialized = self.ensure_initialized()
            if not initialized:
                return None

            if any(self.numba_cuda_path in p for p in path):
                # Re-entrancy - return and carry on
                return None

            oot_path = [
                p.replace(self.numba_path, self.numba_cuda_path) for p in path
            ]
            for finder in sys.meta_path:
                try:
                    spec = finder.find_spec(name, oot_path, target)
                except AttributeError:
                    # Finders written to a pre-Python 3.4 spec for finders will
                    # not implement find_spec. We can skip those altogether.
                    continue
                else:
                    if spec is not None:
                        return spec
