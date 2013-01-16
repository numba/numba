import pickle, sys, os

noseid_file = 'numba/tests/.noseids'
if len(sys.argv) > 1 and sys.argv[1] == '-reset':
    os.remove(noseid_file)
else:
    with open(noseid_file) as fin:
        noseids = pickle.load(fin)

    failed = map(int, noseids['failed'])
    ids = noseids['ids']

    for i in failed:
        print ids[i]


