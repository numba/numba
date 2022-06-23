import re
import pandas as pd


def parse(line):
    _, name, attrs = line.split(' : ')
    assignment = r'=([0-9]+)[,\)]'
    values = [int(x) for x in re.findall(assignment, attrs)]
    regs, shared, local, const, maxthreads = values
    return name, regs, shared, local, const, maxthreads


def read_file(name, col_suffix):
    with open(name) as f:
        entries = [parse(line) for line in sorted(f.readlines())
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
        'name': name,
        f'regs_{col_suffix}': regs,
        f'shared_{col_suffix}': shared,
        f'local_{col_suffix}': local,
        f'const_{col_suffix}': const,
        f'maxthreads_{col_suffix}': maxthreads
    }

    return pd.DataFrame(data=data)


def read_files():
    before_df = read_file('log_without_tid.txt', 'before')
    after_df = read_file('log_with_tid.txt', 'after')

    return before_df, after_df


if __name__ == '__main__':
    read_file('log_with_tid.txt')
