import os

atomic_spirv_path = os.path.join(os.path.dirname(__file__), 'atomic_op.spir')

def read_atomic_spirv_file():
    with open(atomic_spirv_path, 'rb') as fin:
        spirv = fin.read()

    return spirv
