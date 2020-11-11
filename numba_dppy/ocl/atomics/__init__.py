import os
import os.path

def atomic_support_present():
    if os.path.isfile(os.path.join(os.path.dirname(__file__), 'atomic_ops.spir')):
        return True
    else:
        return False

def get_atomic_spirv_path():
    if atomic_support_present():
        return os.path.join(os.path.dirname(__file__), 'atomic_ops.spir')
    else:
        return None

def read_atomic_spirv_file():
    path = get_atomic_spirv_path()
    if path:
        with open(path, 'rb') as fin:
            spirv = fin.read()
        return spirv
    else:
        return None
