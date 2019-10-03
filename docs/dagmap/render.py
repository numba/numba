#!/usr/bin/env python

import os.path
import json
import collections
import yaml
import graphviz
from jinja2 import Environment, FileSystemLoader


Dagmap = collections.namedtuple('Dagmap',
             ['version', 'meta', 'style', 'tasks'])


def parse_yaml(filename):
    with open(filename, 'r') as f:
        contents = yaml.safe_load(f)
    
    meta = contents['meta']
    version = meta['version']
    if version > 1:
        raise Exception('Unsupported version %d' % version)
    del meta['version']

    style = contents['style']
    tasks = contents['tasks']
    if not isinstance(tasks, list):
        raise Exception('"tasks" must be a list')

    return Dagmap(version=version, meta=meta, style=style, tasks=tasks)


def to_graphviz(dagmap):
    G = graphviz.Digraph(format='svg', engine='neato',
        graph_attr=dict(bgcolor="#f4f4f4", pad="0.5", overlap="false"),
        node_attr=dict(width="0.6", style="filled",
                       fillcolor="#83c6de", color="#83c6de", penwidth="3", label="",
                       fontname="helvetica Neue Ultra Light", fontsize="28"),
        edge_attr=dict(color="#616a72", arrowsize="2.0", penwidth="4", fontname="helvetica Neue Ultra Light"))

    G.node(name='_nothing', label='', style='invis')

    for task in dagmap.tasks:
        G.node(name=task['id'], label=task['label'],
               tooltip=task['description'].strip())
        depends_on = task.get('depends_on', ['_nothing'])
        for dep in depends_on:
            if dep == '_nothing':
                attrs = {
                    'style': 'invis',
                }
            else:
                attrs = {}
            G.edge(dep, task['id'], **attrs)

    return G


def main(argv):
    import argparse
    parser = argparse.ArgumentParser(description='Render Dagmap to Graphviz')
    parser.add_argument('-o', '--output', required=True, help='output svg filename')
    parser.add_argument('-t', '--template', default='template.html', help='HTML rendering template')
    parser.add_argument('input', metavar='INPUT', type=str,
                        help='YAML input filename')

    args = parser.parse_args(argv[1:])

    dagmap = parse_yaml(args.input)
    graph = to_graphviz(dagmap)
    svg = graph.pipe().decode('utf-8')

    template_env = Environment(loader=FileSystemLoader(os.path.dirname(__file__)))
    template = template_env.get_template(args.template)
    html = template.render(svg=json.dumps(svg))

    with open(args.output, 'w') as f:
        f.write(html)

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
