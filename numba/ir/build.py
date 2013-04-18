import os
from .generator import build

root = os.path.dirname(os.path.abspath(__file__))
features = ['ast', 'transformer', 'visitor']

def build_normalized():
    fn = os.path.join(root, "Normalized.asdl")
    outdir = os.path.join(root, "normalized")
    build.build_package(fn, features, outdir)

if __name__ == '__main__':
    build_normalized()
