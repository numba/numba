"""
Convert a miniast to an XML document using ElementTree. This allows us to
write XPath unit tests, or just serialize the AST.
"""

try:
    from lxml import etree
    have_lxml = True
except ImportError:
    have_lxml = False
    try:
        # Python 2.5
        from xml.etree import cElementTree as etree
    except ImportError:
        try:
            # Python 2.5
            from xml.etree import ElementTree as etree
        except ImportError:
            try:
                # normal cElementTree install
                import cElementTree as etree
            except ImportError:
                # normal ElementTree install
                import elementtree.ElementTree as etree

import minivisitor

class XMLDumper(minivisitor.PrintTree):

    loop_level = 0

    def visit_FunctionNode(self, node):
        self.treebuilder = etree.TreeBuilder()
        self.visit_Node(node)
        return self.treebuilder.close()

    def start(self, node, attrs={}):
        name = type(node).__name__
        format_value = self.format_value(node)
        if format_value:
            attrs = dict(attrs,
                         value=format_value,
                         id=hex(id(node)),
                         type=node.type)

        attrs = dict((k, str(v)) for k, v in attrs.iteritems())
        self.treebuilder.start(name, attrs)
        return name

    def visit_BinaryOperationNode(self, node):
        name = self.start(node)

        self.treebuilder.start('lhs', {})
        self.visit(node.lhs)
        self.treebuilder.end('lhs')

        self.treebuilder.start('rhs', {})
        self.visit(node.rhs)
        self.treebuilder.end('rhs')

        self.treebuilder.end(name)

    def visit_ForNode(self, node):
        attrs = dict(loop_level=self.loop_level,
                     is_fixup=node.is_fixup,
                     is_controlling_loop=node.is_controlling_loop,
                     is_tiling_loop=node.is_tiling_loop)

        self.loop_level += 1
        self.visit_Node(node, attrs)
        self.loop_level -= 1

    def visit_Node(self, node, attrs={}):
        name = self.start(node, attrs)
        self.visitchildren(node)
        self.treebuilder.end(name)

def tostring(xml_root_element):
    et = etree.ElementTree(xml_root_element)
    kw = {}
    if have_lxml:
        kw['pretty_print'] = True

    return etree.tostring(et, encoding='UTF-8', **kw)