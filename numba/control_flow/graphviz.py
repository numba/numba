# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import subprocess

from numba.control_flow.debug import *
from numba.control_flow.cfstats import NameReference, NameAssignment

class GVContext(object):
    """Graphviz subgraph object."""

    def __init__(self):
        self.blockids = {}
        self.nextid = 0
        self.children = []
        self.sources = {}

    def add(self, child):
        self.children.append(child)

    def nodeid(self, block):
        if block not in self.blockids:
            self.blockids[block] = 'block%d' % self.nextid
            self.nextid += 1
        return self.blockids[block]

    def extract_sources(self, block):
        if not block.positions:
            return ''
        start = min(block.positions)
        stop = max(block.positions)
        srcdescr = start[0]
        if not srcdescr in self.sources:
            self.sources[srcdescr] = list(srcdescr.get_lines())
        lines = self.sources[srcdescr]

        src_descr, begin_line, begin_col = start
        src_descr, end_line, end_col = stop
        lines = lines[begin_line - 1:end_line]
        if not lines:
            return ''
            #lines[0] = lines[0][begin_col:]
        #lines[-1] = lines[-1][:end_col]
        return '\\n'.join([line.strip() for line in lines if line.strip()])

    def render(self, fp, name, annotate_defs=False):
        """Render graphviz dot graph"""
        fp.write('digraph %s {\n' % name)
        fp.write(' node [shape=box];\n')
        for child in self.children:
            child.render(fp, self, annotate_defs)
        fp.write('}\n')

    def escape(self, text):
        return text.replace('"', '\\"').replace('\n', '\\n')


class GV(object):
    """
    Graphviz DOT renderer.
    """

    def __init__(self, name, flow):
        self.name = name
        self.flow = flow

    def format_phis(self, block):
        result = "\\l".join(str(phi) for var, phi in block.phis.iteritems())
        return result

    def render(self, fp, ctx, annotate_defs=False):
        fp.write(' subgraph %s {\n' % self.name)
        for block in self.flow.blocks:
            if block.have_code:
                code = ctx.extract_sources(block)
                if annotate_defs:
                    for stat in block.stats:
                        if isinstance(stat, NameAssignment):
                            code += '\n %s [definition]' % stat.entry.name
                        elif isinstance(stat, NameReference):
                            if stat.entry:
                                code += '\n %s [reference]' % stat.entry.name
            else:
                code = ""

            if block.have_code and block.label == 'empty':
                label = ''
            else:
                label = '%s: ' % block.label

            phis = self.format_phis(block)
            label = '%d\\l%s%s\\n%s' % (block.id, label, phis, code)

            pid = ctx.nodeid(block)
            fp.write('  %s [label="%s"];\n' % (pid, ctx.escape(label)))
        for block in self.flow.blocks:
            pid = ctx.nodeid(block)
            for child in block.children:
                fp.write('  %s -> %s;\n' % (pid, ctx.nodeid(child)))
        fp.write(' }\n')

#----------------------------------------------------------------------------
# Graphviz Rendering
#----------------------------------------------------------------------------

def get_png_output_name(dot_output):
    prefix, ext = os.path.splitext(dot_output)
    i = 0
    while True:
        png_output = "%s%d.png" % (prefix, i)
        if not os.path.exists(png_output):
            break

        i += 1

    return png_output


def write_dotfile(current_directives, dot_output, gv_ctx):
    annotate_defs = current_directives['control_flow.dot_annotate_defs']
    fp = open(dot_output, 'wt')
    try:
        gv_ctx.render(fp, 'module', annotate_defs=annotate_defs)
    finally:
        fp.close()


def write_image(dot_output):
    png_output = get_png_output_name(dot_output)
    fp = open(png_output, 'wb')
    try:
        p = subprocess.Popen(['dot', '-Tpng', dot_output],
                             stdout=fp.fileno(),
                             stderr=subprocess.PIPE)
        p.wait()
    except EnvironmentError as e:
        logger.warn("Unable to write png: %s (did you install the "
                    "'dot' program?). Wrote %s" % (e, dot_output))
    else:
        logger.warn("Wrote %s" % png_output)
    finally:
        fp.close()


def render_gv(node, gv_ctx, flow, current_directives):
    gv_ctx.add(GV(node.name, flow))
    dot_output = current_directives['control_flow.dot_output']
    if dot_output:
        write_dotfile(current_directives, dot_output, gv_ctx)
        write_image(dot_output)
