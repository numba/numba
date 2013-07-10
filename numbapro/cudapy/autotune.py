'''
- Parse jit compile info
- Compute warp occupany histogram
'''


import re

def _sw(s):
    return s.replace(' ', r'\s+')

def _regex(s):
    return re.compile(_sw(s), re.I)

RE_LEAD     = _regex(r'^(?:ptxas )?info : Function properties for ')
RE_REG      = _regex(r'used (?P<num>\d+) registers')
RE_STACK    = _regex(r'(?P<num>\d+) (?:bytes )?stack')
RE_SHARED   = _regex(r'(?P<num>\d+) bytes smem')
RE_LOCAL    = _regex(r'(?P<num>\d+) bytes lmem')

def parse_compile_info(text):
    return dict(gen_parse_compile_info(text))

def gen_parse_compile_info(text):
    '''Generator that returns function (name, resource dict)
        
    May yield the same function more than once.  
    In that case, the latter should replace the prior.
    
    Usage:
    
    >>> dict(parse_compile_info(compile_info))
    
    '''
    lines = text.splitlines()
    readline = iter(lines).next

    try:
        ln = readline()
        while True:
            m = RE_LEAD.match(ln)
            if not m:
                # not a lead line; continue
                ln = readline()
                continue
            # start parsing information
            remaining = ln[len(m.group(0)):]
            # function name
            fname = parse_function_name(remaining)
            # resource info
            ln = readline()
            resources = {}
            try:
                while True:
                    more = False
                    for section in ln.split(','):
                        res = parse_resources(section)
                        if res:
                            k, v = res
                            resources[k] = v
                            more = True
                    ln = readline()
                    if not more:
                        break
            finally:
                yield fname, resources
    except StopIteration:
        pass

def parse_function_name(text):
    name = text.strip().rstrip(':')
    # has quote?
    if name.startswith("'"):
        assert name.endswith("'")
        name = name[1:-1]
    return name

def parse_resources(text):
    '''
    Returns (key, value) tuple on successful parse;
            otherwise, None
    '''
    relst = [('reg',    RE_REG),
             ('stack',  RE_STACK),
             ('shared', RE_SHARED),
             ('local',  RE_LOCAL)]
    for resname, regex in relst:
        m = regex.search(text)
        if m:
            key = resname
            val = int(m.group('num'))
            return key, val

