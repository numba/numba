import os
from .generator import build

root = os.path.dirname(os.path.abspath(__file__))
features = ['ast', 'transformer', 'visitor']

def build_normalized():
    fn = os.path.join(root, "Normalized.asdl")
    outdir = os.path.join(root, "normalized")
    build.build_package(fn, features, outdir)

def build_untyped():
    fn = os.path.join(root, "UntypedIR.asdl")
    outdir = os.path.join(root, "untyped")
    build.build_package(fn, features, outdir)

def build_typed():
    fn = os.path.join(root, "TypedIR.asdl")
    outdir = os.path.join(root, "typed")
    build.build_package(fn, features, outdir)


if __name__ == '__main__':
    build_normalized()
    build_untyped()
    # build_typed()