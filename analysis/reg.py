import re
import pandas as pd


def parse(line):
    _, name, attrs = line.split(' : ')
    assignment = r'=([0-9]+)[,\)]'
    values = [int(x) for x in re.findall(assignment, attrs)]
    regs, shared, local, const, maxthreads = values
    return name, regs, shared, local, const, maxthreads


def read_file(name):
    with open(name) as f:
        entries = [parse(line) for line in f.readlines()
                   if 'ATTRIBUTES' in line]
    print(f'Total entries: {len(entries)}')

    name = []
    regs = []
    shared = []
    local = []
    const = []
    maxthreads = []

    for values in entries:
        name.append(values[0])
        regs.append(values[1])
        shared.append(values[2])
        local.append(values[3])
        const.append(values[4])
        maxthreads.append(values[5])

    data = {
        #'name': name,
        'regs': regs,
        'shared': shared,
        'local': local,
        'const': const,
        'maxthreads': maxthreads
    }

    return pd.DataFrame(data=data, index=name)


if __name__ == '__main__':
    read_file('log_with_tid.txt')
