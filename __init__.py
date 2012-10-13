import os

root = os.path.dirname(os.path.abspath(__file__))

def get_include():
    return os.path.join(root, 'include')
