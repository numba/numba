# -*- coding: Latin-1 -*-
"""Graphviz's dot language Python interface.

This module provides with a full interface to create handle modify
and process graphs in Graphviz's dot language.

References:

pydot Homepage: http://code.google.com/p/pydot/
Graphviz:       http://www.graphviz.org/
DOT Language:   http://www.graphviz.org/doc/info/lang.html

Programmed and tested with Graphviz 2.26.3 and Python 2.6 on OSX 10.6.4

Copyright (c) 2005-2011 Ero Carrera <ero.carrera@gmail.com>

Distributed under MIT license [http://opensource.org/licenses/mit-license.html].
"""

__revision__ = "$LastChangedRevision: 27 $"
__author__ = 'Ero Carrera'
__version__ = '1.0.%d' % int( __revision__[21:-2] )
__license__ = 'MIT'

import os
import re
import subprocess
import tempfile
import copy
try:
    import dot_parser
except Exception, e:
    # print "Couldn't import dot_parser, loading of dot files will not be possible."
    pass



GRAPH_ATTRIBUTES = set( ['Damping', 'K', 'URL', 'aspect', 'bb', 'bgcolor',
    'center', 'charset', 'clusterrank', 'colorscheme', 'comment', 'compound',
    'concentrate', 'defaultdist', 'dim', 'dimen', 'diredgeconstraints',
    'dpi', 'epsilon', 'esep', 'fontcolor', 'fontname', 'fontnames',
    'fontpath', 'fontsize', 'id', 'label', 'labeljust', 'labelloc',
    'landscape', 'layers', 'layersep', 'layout', 'levels', 'levelsgap',
    'lheight', 'lp', 'lwidth', 'margin', 'maxiter', 'mclimit', 'mindist',
    'mode', 'model', 'mosek', 'nodesep', 'nojustify', 'normalize', 'nslimit',
    'nslimit1', 'ordering', 'orientation', 'outputorder', 'overlap',
    'overlap_scaling', 'pack', 'packmode', 'pad', 'page', 'pagedir',
    'quadtree', 'quantum', 'rankdir', 'ranksep', 'ratio', 'remincross',
    'repulsiveforce', 'resolution', 'root', 'rotate', 'searchsize', 'sep',
    'showboxes', 'size', 'smoothing', 'sortv', 'splines', 'start',
    'stylesheet', 'target', 'truecolor', 'viewport', 'voro_margin',
    # for subgraphs
    'rank' ] )


EDGE_ATTRIBUTES = set( ['URL', 'arrowhead', 'arrowsize', 'arrowtail',
    'color', 'colorscheme', 'comment', 'constraint', 'decorate', 'dir',
    'edgeURL', 'edgehref', 'edgetarget', 'edgetooltip', 'fontcolor',
    'fontname', 'fontsize', 'headURL', 'headclip', 'headhref', 'headlabel',
    'headport', 'headtarget', 'headtooltip', 'href', 'id', 'label',
    'labelURL', 'labelangle', 'labeldistance', 'labelfloat', 'labelfontcolor',
    'labelfontname', 'labelfontsize', 'labelhref', 'labeltarget',
    'labeltooltip', 'layer', 'len', 'lhead', 'lp', 'ltail', 'minlen',
    'nojustify', 'penwidth', 'pos', 'samehead', 'sametail', 'showboxes',
    'style', 'tailURL', 'tailclip', 'tailhref', 'taillabel', 'tailport',
    'tailtarget', 'tailtooltip', 'target', 'tooltip', 'weight',
    'rank' ] )


NODE_ATTRIBUTES = set( ['URL', 'color', 'colorscheme', 'comment',
    'distortion', 'fillcolor', 'fixedsize', 'fontcolor', 'fontname',
    'fontsize', 'group', 'height', 'id', 'image', 'imagescale', 'label',
    'labelloc', 'layer', 'margin', 'nojustify', 'orientation', 'penwidth',
    'peripheries', 'pin', 'pos', 'rects', 'regular', 'root', 'samplepoints',
    'shape', 'shapefile', 'showboxes', 'sides', 'skew', 'sortv', 'style',
    'target', 'tooltip', 'vertices', 'width', 'z',
    # The following are attributes dot2tex
    'texlbl',  'texmode' ] )


CLUSTER_ATTRIBUTES = set( ['K', 'URL', 'bgcolor', 'color', 'colorscheme',
    'fillcolor', 'fontcolor', 'fontname', 'fontsize', 'label', 'labeljust',
    'labelloc', 'lheight', 'lp', 'lwidth', 'nojustify', 'pencolor',
    'penwidth', 'peripheries', 'sortv', 'style', 'target', 'tooltip'] )


#
# Extented version of ASPN's Python Cookbook Recipe:
# Frozen dictionaries.
# http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/414283
#
# This version freezes dictionaries used as values within dictionaries.
#
class frozendict(dict):
    def _blocked_attribute(obj):
        raise AttributeError, "A frozendict cannot be modified."
    _blocked_attribute = property(_blocked_attribute)

    __delitem__ = __setitem__ = clear = _blocked_attribute
    pop = popitem = setdefault = update = _blocked_attribute

    def __new__(cls, *args, **kw):
        new = dict.__new__(cls)

        args_ = []
        for arg in args:
            if isinstance(arg, dict):
                arg = copy.copy(arg)
                for k, v in arg.iteritems():
                    if isinstance(v, frozendict):
                        arg[k] = v
                    elif isinstance(v, dict):
                        arg[k] = frozendict(v)
                    elif isinstance(v, list):
                        v_ = list()
                        for elm in v:
                            if isinstance(elm, dict):
                                v_.append( frozendict(elm) )
                            else:
                                v_.append( elm )
                        arg[k] = tuple(v_)
                args_.append( arg )
            else:
                args_.append( arg )

        dict.__init__(new, *args_, **kw)
        return new

    def __init__(self, *args, **kw):
        pass

    def __hash__(self):
        try:
            return self._cached_hash
        except AttributeError:
            h = self._cached_hash = hash(tuple(sorted(self.iteritems())))
            return h

    def __repr__(self):
        return "frozendict(%s)" % dict.__repr__(self)


dot_keywords = ['graph', 'subgraph', 'digraph', 'node', 'edge', 'strict']

id_re_alpha_nums = re.compile('^[_a-zA-Z][a-zA-Z0-9_,]*$', re.UNICODE)
id_re_alpha_nums_with_ports = re.compile('^[_a-zA-Z][a-zA-Z0-9_,:\"]*[a-zA-Z0-9_,\"]+$', re.UNICODE)
id_re_num = re.compile('^[0-9,]+$', re.UNICODE)
id_re_with_port = re.compile('^([^:]*):([^:]*)$', re.UNICODE)
id_re_dbl_quoted = re.compile('^\".*\"$', re.S|re.UNICODE)
id_re_html = re.compile('^<.*>$', re.S|re.UNICODE)


def needs_quotes( s ):
    """Checks whether a string is a dot language ID.

    It will check whether the string is solely composed
    by the characters allowed in an ID or not.
    If the string is one of the reserved keywords it will
    need quotes too but the user will need to add them
    manually.
    """

    # If the name is a reserved keyword it will need quotes but pydot
    # can't tell when it's being used as a keyword or when it's simply
    # a name. Hence the user needs to supply the quotes when an element
    # would use a reserved keyword as name. This function will return
    # false indicating that a keyword string, if provided as-is, won't
    # need quotes.
    if s in dot_keywords:
        return False

    chars = [ord(c) for c in s if ord(c)>0x7f or ord(c)==0]
    if chars and not id_re_dbl_quoted.match(s) and not id_re_html.match(s):
        return True

    for test_re in [id_re_alpha_nums, id_re_num, id_re_dbl_quoted, id_re_html, id_re_alpha_nums_with_ports]:
        if test_re.match(s):
            return False

    m = id_re_with_port.match(s)
    if m:
        return needs_quotes(m.group(1)) or needs_quotes(m.group(2))

    return True


def quote_if_necessary(s):

    if isinstance(s, bool):
        if s is True:
            return 'True'
        return 'False'

    if not isinstance( s, basestring ):
        return s

    if not s:
        return s

    if needs_quotes(s):
        replace = {'"'  : r'\"',
                   "\n" : r'\n',
                   "\r" : r'\r'}
        for (a,b) in replace.items():
            s = s.replace(a, b)

        return '"' + s + '"'

    return s



def graph_from_dot_data(data):
    """Load graph as defined by data in DOT format.

    The data is assumed to be in DOT format. It will
    be parsed and a Dot class will be returned,
    representing the graph.
    """

    return dot_parser.parse_dot_data(data)


def graph_from_dot_file(path):
    """Load graph as defined by a DOT file.

    The file is assumed to be in DOT format. It will
    be loaded, parsed and a Dot class will be returned,
    representing the graph.
    """

    fd = file(path, 'rb')
    data = fd.read()
    fd.close()

    return graph_from_dot_data(data)



def graph_from_edges(edge_list, node_prefix='', directed=False):
    """Creates a basic graph out of an edge list.

    The edge list has to be a list of tuples representing
    the nodes connected by the edge.
    The values can be anything: bool, int, float, str.

    If the graph is undirected by default, it is only
    calculated from one of the symmetric halves of the matrix.
    """

    if directed:
        graph = Dot(graph_type='digraph')

    else:
        graph = Dot(graph_type='graph')

    for edge in edge_list:

        if isinstance(edge[0], str):
            src = node_prefix + edge[0]
        else:
            src = node_prefix + str(edge[0])

        if isinstance(edge[1], str):
            dst = node_prefix + edge[1]
        else:
            dst = node_prefix + str(edge[1])

        e = Edge( src, dst )
        graph.add_edge(e)

    return graph


def graph_from_adjacency_matrix(matrix, node_prefix= u'', directed=False):
    """Creates a basic graph out of an adjacency matrix.

    The matrix has to be a list of rows of values
    representing an adjacency matrix.
    The values can be anything: bool, int, float, as long
    as they can evaluate to True or False.
    """

    node_orig = 1

    if directed:
        graph = Dot(graph_type='digraph')
    else:
        graph = Dot(graph_type='graph')

    for row in matrix:
        if not directed:
            skip = matrix.index(row)
            r = row[skip:]
        else:
            skip = 0
            r = row
        node_dest = skip+1

        for e in r:
            if e:
                graph.add_edge(
                    Edge( node_prefix + node_orig,
                        node_prefix + node_dest) )
            node_dest += 1
        node_orig += 1

    return graph



def graph_from_incidence_matrix(matrix, node_prefix='', directed=False):
    """Creates a basic graph out of an incidence matrix.

    The matrix has to be a list of rows of values
    representing an incidence matrix.
    The values can be anything: bool, int, float, as long
    as they can evaluate to True or False.
    """

    node_orig = 1

    if directed:
        graph = Dot(graph_type='digraph')
    else:
        graph = Dot(graph_type='graph')

    for row in matrix:
        nodes = []
        c = 1

        for node in row:
            if node:
                nodes.append(c*node)
            c += 1
            nodes.sort()

        if len(nodes) == 2:
            graph.add_edge(
                Edge( node_prefix + abs(nodes[0]),
                    node_prefix + nodes[1] ))

    if not directed:
        graph.set_simplify(True)

    return graph




def __find_executables(path):
    """Used by find_graphviz

    path - single directory as a string

    If any of the executables are found, it will return a dictionary
    containing the program names as keys and their paths as values.

    Otherwise returns None
    """

    success = False
    progs = {'dot': '', 'twopi': '', 'neato': '', 'circo': '', 'fdp': '', 'sfdp': ''}

    was_quoted = False
    path = path.strip()
    if path.startswith('"') and path.endswith('"'):
        path = path[1:-1]
        was_quoted =  True

    if os.path.isdir(path) :

        for prg in progs.iterkeys():

            if progs[prg]:
                continue

            if os.path.exists( os.path.join(path, prg) ):

                if was_quoted:
                    progs[prg] = '"' + os.path.join(path, prg) + '"'
                else:
                    progs[prg] = os.path.join(path, prg)

                success = True

            elif os.path.exists( os.path.join(path, prg + '.exe') ):

                if was_quoted:
                    progs[prg] = '"' + os.path.join(path, prg + '.exe') + '"'
                else:
                    progs[prg] = os.path.join(path, prg + '.exe')

                success = True

    if success:

        return progs

    else:

        return None



# The multi-platform version of this 'find_graphviz' function was
# contributed by Peter Cock
#
def find_graphviz():
    """Locate Graphviz's executables in the system.

    Tries three methods:

    First: Windows Registry (Windows only)
    This requires Mark Hammond's pywin32 is installed.

    Secondly: Search the path
    It will look for 'dot', 'twopi' and 'neato' in all the directories
    specified in the PATH environment variable.

    Thirdly: Default install location (Windows only)
    It will look for 'dot', 'twopi' and 'neato' in the default install
    location under the "Program Files" directory.

    It will return a dictionary containing the program names as keys
    and their paths as values.

    If this fails, it returns None.
    """

    # Method 1 (Windows only)
    #
    if os.sys.platform == 'win32':

        HKEY_LOCAL_MACHINE =    0x80000002
        KEY_QUERY_VALUE =       0x0001

        RegOpenKeyEx = None
        RegQueryValueEx = None
        RegCloseKey = None

        try:
            import win32api, win32con
            RegOpenKeyEx = win32api.RegOpenKeyEx
            RegQueryValueEx = win32api.RegQueryValueEx
            RegCloseKey = win32api.RegCloseKey

        except ImportError:
            # Print a messaged suggesting they install these?
            #
            pass

        try:
            import ctypes

            def RegOpenKeyEx(key, subkey, opt, sam):
                result = ctypes.c_uint(0)
                ctypes.windll.advapi32.RegOpenKeyExA(key, subkey, opt, sam, ctypes.byref(result))
                return result.value

            def RegQueryValueEx( hkey, valuename ):
                data_type = ctypes.c_uint(0)
                data_len = ctypes.c_uint(1024)
                data = ctypes.create_string_buffer( 1024 )

                res = ctypes.windll.advapi32.RegQueryValueExA(hkey, valuename, 0,
                    ctypes.byref(data_type), data, ctypes.byref(data_len))

                return data.value

            RegCloseKey = ctypes.windll.advapi32.RegCloseKey

        except ImportError:
            # Print a messaged suggesting they install these?
            #
            pass

        if RegOpenKeyEx is not None:

            # Get the GraphViz install path from the registry
            #
            hkey = None
            potentialKeys = [
                "SOFTWARE\\ATT\\Graphviz",
                "SOFTWARE\\AT&T Research Labs\\Graphviz",
            ]
            for potentialKey in potentialKeys:

                try:
                    hkey = RegOpenKeyEx( HKEY_LOCAL_MACHINE,
                        potentialKey, 0, KEY_QUERY_VALUE )

                    if hkey is not None:
                        path = RegQueryValueEx( hkey, "InstallPath" )
                        RegCloseKey( hkey )

                        # The regitry variable might exist, left by old installations
                        # but with no value, in those cases we keep searching...
                        if not path:
                            continue

                        # Now append the "bin" subdirectory:
                        #
                        path = os.path.join(path, "bin")
                        progs = __find_executables(path)
                        if progs is not None :
                            #print "Used Windows registry"
                            return progs

                except Exception, excp:
                    #raise excp
                    pass
                else:
                    break



    # Method 2 (Linux, Windows etc)
    #
    if os.environ.has_key('PATH'):

        for path in os.environ['PATH'].split(os.pathsep):
            progs = __find_executables(path)
            if progs is not None :
                #print "Used path"
                return progs

    # Method 3 (Windows only)
    #
    if os.sys.platform == 'win32':

        # Try and work out the equivalent of "C:\Program Files" on this
        # machine (might be on drive D:, or in a different language)
        #

        if os.environ.has_key('PROGRAMFILES'):

            # Note, we could also use the win32api to get this
            # information, but win32api may not be installed.

            path = os.path.join(os.environ['PROGRAMFILES'], 'ATT', 'GraphViz', 'bin')

        else:

            #Just in case, try the default...
            path = r"C:\Program Files\att\Graphviz\bin"

        progs = __find_executables(path)

        if progs is not None :

            #print "Used default install location"
            return progs


    for path in (
        '/usr/bin', '/usr/local/bin',
        '/opt/local/bin',
        '/opt/bin', '/sw/bin', '/usr/share',
        '/Applications/Graphviz.app/Contents/MacOS/' ):

        progs = __find_executables(path)
        if progs is not None :
            #print "Used path"
            return progs

    # Failed to find GraphViz
    #
    return None


class Common:
    """Common information to several classes.

    Should not be directly used, several classes are derived from
    this one.
    """


    def __getstate__(self):

        dict = copy.copy(self.obj_dict)

        return dict


    def __setstate__(self, state):

        self.obj_dict = state


    def __get_attribute__(self, attr):
        """Look for default attributes for this node"""

        attr_val = self.obj_dict['attributes'].get(attr, None)

        if attr_val is None:
            # get the defaults for nodes/edges

            default_node_name = self.obj_dict['type']

            # The defaults for graphs are set on a node named 'graph'
            if default_node_name in ('subgraph', 'digraph', 'cluster'):
                default_node_name = 'graph'

            g = self.get_parent_graph()
            if g is not None:
                defaults = g.get_node( default_node_name )
            else:
                return None

            # Multiple defaults could be set by having repeated 'graph [...]'
            # 'node [...]', 'edge [...]' statements. In such case, if the
            # same attribute is set in different statements, only the first
            # will be returned. In order to get all, one would call the
            # get_*_defaults() methods and handle those. Or go node by node
            # (of the ones specifying defaults) and modify the attributes
            # individually.
            #
            if not isinstance(defaults, (list, tuple)):
                defaults = [defaults]

            for default in defaults:
                attr_val = default.obj_dict['attributes'].get(attr, None)
                if attr_val:
                    return attr_val
        else:
            return attr_val

        return None


    def set_parent_graph(self, parent_graph):

        self.obj_dict['parent_graph'] = parent_graph


    def get_parent_graph(self):

        return self.obj_dict.get('parent_graph', None)


    def set(self, name, value):
        """Set an attribute value by name.

        Given an attribute 'name' it will set its value to 'value'.
        There's always the possibility of using the methods:

            set_'name'(value)

        which are defined for all the existing attributes.
        """

        self.obj_dict['attributes'][name] = value


    def get(self, name):
        """Get an attribute value by name.

        Given an attribute 'name' it will get its value.
        There's always the possibility of using the methods:

            get_'name'()

        which are defined for all the existing attributes.
        """

        return self.obj_dict['attributes'].get(name, None)


    def get_attributes(self):
        """"""

        return self.obj_dict['attributes']


    def set_sequence(self, seq):

        self.obj_dict['sequence'] = seq


    def get_sequence(self):

        return self.obj_dict['sequence']


    def create_attribute_methods(self, obj_attributes):

        #for attr in self.obj_dict['attributes']:
        for attr in obj_attributes:

            # Generate all the Setter methods.
            #
            self.__setattr__( 'set_'+attr, lambda x, a=attr : self.obj_dict['attributes'].__setitem__(a, x) )

            # Generate all the Getter methods.
            #
            self.__setattr__('get_'+attr, lambda a=attr : self.__get_attribute__(a))



class Error(Exception):
    """General error handling class.
    """
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return self.value


class InvocationException(Exception):
    """To indicate that a ploblem occurred while running any of the GraphViz executables.
    """
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return self.value



class Node(object, Common):
    """A graph node.

    This class represents a graph's node with all its attributes.

    node(name, attribute=value, ...)

    name: node's name

    All the attributes defined in the Graphviz dot language should
    be supported.
    """

    def __init__(self, name = '', obj_dict = None, **attrs):

        #
        # Nodes will take attributes of all other types because the defaults
        # for any GraphViz object are dealt with as if they were Node definitions
        #

        if obj_dict is not None:

            self.obj_dict = obj_dict

        else:

            self.obj_dict = dict()

            # Copy the attributes
            #
            self.obj_dict[ 'attributes' ] = dict( attrs )
            self.obj_dict[ 'type' ] = 'node'
            self.obj_dict[ 'parent_graph' ] = None
            self.obj_dict[ 'parent_node_list' ] = None
            self.obj_dict[ 'sequence' ] = None

            # Remove the compass point
            #
            port = None
            if isinstance(name, basestring) and not name.startswith('"'):
                idx = name.find(':')
                if idx > 0 and idx+1 < len(name):
                    name, port = name[:idx], name[idx:]

            if isinstance(name, (long, int)):
                name = str(name)

            self.obj_dict['name'] = quote_if_necessary( name )
            self.obj_dict['port'] = port

        self.create_attribute_methods(NODE_ATTRIBUTES)



    def set_name(self, node_name):
        """Set the node's name."""

        self.obj_dict['name'] = node_name


    def get_name(self):
        """Get the node's name."""

        return self.obj_dict['name']


    def get_port(self):
        """Get the node's port."""

        return self.obj_dict['port']


    def add_style(self, style):

        styles = self.obj_dict['attributes'].get('style', None)
        if not styles and style:
            styles = [ style ]
        else:
            styles = styles.split(',')
            styles.append( style )

        self.obj_dict['attributes']['style'] = ','.join( styles )


    def to_string(self):
        """Returns a string representation of the node in dot language.
        """


        # RMF: special case defaults for node, edge and graph properties.
        #
        node = quote_if_necessary(self.obj_dict['name'])

        node_attr = list()

        for attr, value in self.obj_dict['attributes'].iteritems():
            if value is not None:
                node_attr.append( '%s=%s' % (attr, quote_if_necessary(value) ) )
            else:
                node_attr.append( attr )


        # No point in having nodes setting any defaults if the don't set
        # any attributes...
        #
        if node in ('graph', 'node', 'edge') and len(node_attr) == 0:
            return ''

        node_attr = ', '.join(node_attr)

        if node_attr:
            node += ' [' + node_attr + ']'

        return node + ';'



class Edge(object,  Common ):
    """A graph edge.

    This class represents a graph's edge with all its attributes.

    edge(src, dst, attribute=value, ...)

    src: source node's name
    dst: destination node's name

    All the attributes defined in the Graphviz dot language should
    be supported.

 	Attributes can be set through the dynamically generated methods:

     set_[attribute name], i.e. set_label, set_fontname

    or directly by using the instance's special dictionary:

     Edge.obj_dict['attributes'][attribute name], i.e.

        edge_instance.obj_dict['attributes']['label']
        edge_instance.obj_dict['attributes']['fontname']

    """



    def __init__(self, src='', dst='', obj_dict=None, **attrs):

        if isinstance(src, (list, tuple)) and dst == '':
            src, dst = src

        if obj_dict is not None:

            self.obj_dict = obj_dict

        else:

            self.obj_dict = dict()

            # Copy the attributes
            #
            self.obj_dict[ 'attributes' ] = dict( attrs )
            self.obj_dict[ 'type' ] = 'edge'
            self.obj_dict[ 'parent_graph' ] = None
            self.obj_dict[ 'parent_edge_list' ] = None
            self.obj_dict[ 'sequence' ] = None

            if isinstance(src, Node):
                src = src.get_name()

            if isinstance(dst, Node):
                dst = dst.get_name()

            points = ( quote_if_necessary( src) , quote_if_necessary( dst) )

            self.obj_dict['points'] = points

        self.create_attribute_methods(EDGE_ATTRIBUTES)


    def get_source(self):
        """Get the edges source node name."""

        return self.obj_dict['points'][0]


    def get_destination(self):
        """Get the edge's destination node name."""

        return self.obj_dict['points'][1]


    def __hash__(self):

         return hash( hash(self.get_source()) + hash(self.get_destination()) )


    def __eq__(self, edge):
        """Compare two edges.

        If the parent graph is directed, arcs linking
        node A to B are considered equal and A->B != B->A

        If the parent graph is undirected, any edge
        connecting two nodes is equal to any other
        edge connecting the same nodes, A->B == B->A
        """

        if not isinstance(edge, Edge):
            raise Error, "Can't compare and edge to a non-edge object."

        if self.get_parent_graph().get_top_graph_type() == 'graph':

            # If the graph is undirected, the edge has neither
            # source nor destination.
            #
            if	( ( self.get_source() == edge.get_source() and self.get_destination() == edge.get_destination() ) or
                ( edge.get_source() == self.get_destination() and edge.get_destination() == self.get_source() ) ):
                return True

        else:

            if self.get_source()==edge.get_source() and self.get_destination()==edge.get_destination() :
                return True

        return False



    def parse_node_ref(self, node_str):

        if not isinstance(node_str, str):
            return node_str

        if node_str.startswith('"') and node_str.endswith('"'):

            return node_str

        node_port_idx = node_str.rfind(':')

        if node_port_idx>0 and node_str[0]=='"' and node_str[node_port_idx-1]=='"':

            return node_str

        if node_port_idx>0:

            a = node_str[:node_port_idx]
            b = node_str[node_port_idx+1:]

            node = quote_if_necessary(a)

            node += ':'+quote_if_necessary(b)

            return node

        return node_str


    def to_string(self):
        """Returns a string representation of the edge in dot language.
        """

        src = self.parse_node_ref( self.get_source() )
        dst = self.parse_node_ref( self.get_destination() )

        if isinstance(src, frozendict):
            edge = [ Subgraph(obj_dict=src).to_string() ]
        elif isinstance(src, (int, long)):
            edge = [ str(src) ]
        else:
            edge = [ src ]

        if	(self.get_parent_graph() and
            self.get_parent_graph().get_top_graph_type() and
            self.get_parent_graph().get_top_graph_type() == 'digraph' ):

            edge.append( '->' )

        else:
            edge.append( '--' )

        if isinstance(dst, frozendict):
            edge.append( Subgraph(obj_dict=dst).to_string() )
        elif isinstance(dst, (int, long)):
            edge.append( str(dst) )
        else:
            edge.append( dst )


        edge_attr = list()

        for attr, value in self.obj_dict['attributes'].iteritems():

            if value is not None:
                edge_attr.append( '%s=%s' % (attr, quote_if_necessary(value) ) )
            else:
                edge_attr.append( attr )

        edge_attr = ', '.join(edge_attr)

        if edge_attr:
            edge.append( ' [' + edge_attr + ']' )

        return ' '.join(edge) + ';'





class Graph(object, Common):
    """Class representing a graph in Graphviz's dot language.

    This class implements the methods to work on a representation
    of a graph in Graphviz's dot language.

    graph(  graph_name='G', graph_type='digraph',
        strict=False, suppress_disconnected=False, attribute=value, ...)

    graph_name:
        the graph's name
    graph_type:
        can be 'graph' or 'digraph'
    suppress_disconnected:
        defaults to False, which will remove from the
        graph any disconnected nodes.
    simplify:
        if True it will avoid displaying equal edges, i.e.
        only one edge between two nodes. removing the
        duplicated ones.

    All the attributes defined in the Graphviz dot language should
    be supported.

    Attributes can be set through the dynamically generated methods:

     set_[attribute name], i.e. set_size, set_fontname

    or using the instance's attributes:

     Graph.obj_dict['attributes'][attribute name], i.e.

        graph_instance.obj_dict['attributes']['label']
        graph_instance.obj_dict['attributes']['fontname']
    """


    def __init__(self, graph_name='G', obj_dict=None, graph_type='digraph', strict=False,
        suppress_disconnected=False, simplify=False, **attrs):

        if obj_dict is not None:
            self.obj_dict = obj_dict

        else:

            self.obj_dict = dict()

            self.obj_dict['attributes'] = dict(attrs)

            if graph_type not in ['graph', 'digraph']:
                raise Error, 'Invalid type "%s". Accepted graph types are: graph, digraph, subgraph' % graph_type


            self.obj_dict['name'] = quote_if_necessary(graph_name)
            self.obj_dict['type'] = graph_type

            self.obj_dict['strict'] = strict
            self.obj_dict['suppress_disconnected'] = suppress_disconnected
            self.obj_dict['simplify'] = simplify

            self.obj_dict['current_child_sequence'] = 1
            self.obj_dict['nodes'] = dict()
            self.obj_dict['edges'] = dict()
            self.obj_dict['subgraphs'] = dict()

            self.set_parent_graph(self)


        self.create_attribute_methods(GRAPH_ATTRIBUTES)


    def get_graph_type(self):

        return self.obj_dict['type']


    def get_top_graph_type(self):

        parent = self
        while True:
            parent_ = parent.get_parent_graph()
            if parent_ == parent:
                break
            parent = parent_

        return parent.obj_dict['type']


    def set_graph_defaults(self, **attrs):

        self.add_node( Node('graph', **attrs) )


    def get_graph_defaults(self, **attrs):

        graph_nodes = self.get_node('graph')

        if isinstance( graph_nodes, (list, tuple)):
            return [ node.get_attributes() for node in graph_nodes ]

        return graph_nodes.get_attributes()



    def set_node_defaults(self, **attrs):

        self.add_node( Node('node', **attrs) )


    def get_node_defaults(self, **attrs):


        graph_nodes = self.get_node('node')

        if isinstance( graph_nodes, (list, tuple)):
            return [ node.get_attributes() for node in graph_nodes ]

        return graph_nodes.get_attributes()


    def set_edge_defaults(self, **attrs):

        self.add_node( Node('edge', **attrs) )



    def get_edge_defaults(self, **attrs):

        graph_nodes = self.get_node('edge')

        if isinstance( graph_nodes, (list, tuple)):
            return [ node.get_attributes() for node in graph_nodes ]

        return graph_nodes.get_attributes()



    def set_simplify(self, simplify):
        """Set whether to simplify or not.

        If True it will avoid displaying equal edges, i.e.
        only one edge between two nodes. removing the
        duplicated ones.
        """

        self.obj_dict['simplify'] = simplify



    def get_simplify(self):
        """Get whether to simplify or not.

        Refer to set_simplify for more information.
        """

        return self.obj_dict['simplify']


    def set_type(self, graph_type):
        """Set the graph's type, 'graph' or 'digraph'."""

        self.obj_dict['type'] = graph_type



    def get_type(self):
        """Get the graph's type, 'graph' or 'digraph'."""

        return self.obj_dict['type']



    def set_name(self, graph_name):
        """Set the graph's name."""

        self.obj_dict['name'] = graph_name



    def get_name(self):
        """Get the graph's name."""

        return self.obj_dict['name']



    def set_strict(self, val):
        """Set graph to 'strict' mode.

        This option is only valid for top level graphs.
        """

        self.obj_dict['strict'] = val



    def get_strict(self, val):
        """Get graph's 'strict' mode (True, False).

        This option is only valid for top level graphs.
        """

        return self.obj_dict['strict']



    def set_suppress_disconnected(self, val):
        """Suppress disconnected nodes in the output graph.

        This option will skip nodes in the graph with no incoming or outgoing
        edges. This option works also for subgraphs and has effect only in the
        current graph/subgraph.
        """

        self.obj_dict['suppress_disconnected'] = val



    def get_suppress_disconnected(self, val):
        """Get if suppress disconnected is set.

        Refer to set_suppress_disconnected for more information.
        """

        return self.obj_dict['suppress_disconnected']


    def get_next_sequence_number(self):

        seq = self.obj_dict['current_child_sequence']

        self.obj_dict['current_child_sequence'] += 1

        return seq



    def add_node(self, graph_node):
        """Adds a node object to the graph.

        It takes a node object as its only argument and returns
        None.
        """

        if not isinstance(graph_node, Node):
            raise TypeError('add_node() received a non node class object: ' + str(graph_node))


        node = self.get_node(graph_node.get_name())

        if not node:

            self.obj_dict['nodes'][graph_node.get_name()] = [ graph_node.obj_dict ]

            #self.node_dict[graph_node.get_name()] = graph_node.attributes
            graph_node.set_parent_graph(self.get_parent_graph())

        else:

            self.obj_dict['nodes'][graph_node.get_name()].append( graph_node.obj_dict )

        graph_node.set_sequence(self.get_next_sequence_number())



    def del_node(self, name, index=None):
        """Delete a node from the graph.

        Given a node's name all node(s) with that same name
        will be deleted if 'index' is not specified or set
        to None.
        If there are several nodes with that same name and
        'index' is given, only the node in that position
        will be deleted.

        'index' should be an integer specifying the position
        of the node to delete. If index is larger than the
        number of nodes with that name, no action is taken.

        If nodes are deleted it returns True. If no action
        is taken it returns False.
        """

        if isinstance(name, Node):
            name = name.get_name()

        if self.obj_dict['nodes'].has_key(name):

            if index is not None and index < len(self.obj_dict['nodes'][name]):
                del self.obj_dict['nodes'][name][index]
                return True
            else:
                del self.obj_dict['nodes'][name]
                return True

        return False


    def get_node(self, name):
        """Retrieve a node from the graph.

        Given a node's name the corresponding Node
        instance will be returned.

        If one or more nodes exist with that name a list of
        Node instances is returned.
        An empty list is returned otherwise.
        """

        match = list()

        if self.obj_dict['nodes'].has_key(name):

            match.extend( [ Node( obj_dict = obj_dict ) for obj_dict in self.obj_dict['nodes'][name] ])

        return match


    def get_nodes(self):
        """Get the list of Node instances."""

        return self.get_node_list()


    def get_node_list(self):
        """Get the list of Node instances.

        This method returns the list of Node instances
        composing the graph.
        """

        node_objs = list()

        for node, obj_dict_list in self.obj_dict['nodes'].iteritems():
                node_objs.extend( [ Node( obj_dict = obj_d ) for obj_d in obj_dict_list ] )

        return node_objs



    def add_edge(self, graph_edge):
        """Adds an edge object to the graph.

        It takes a edge object as its only argument and returns
        None.
        """

        if not isinstance(graph_edge, Edge):
            raise TypeError('add_edge() received a non edge class object: ' + str(graph_edge))

        edge_points = ( graph_edge.get_source(), graph_edge.get_destination() )

        if self.obj_dict['edges'].has_key(edge_points):

            edge_list = self.obj_dict['edges'][edge_points]
            edge_list.append(graph_edge.obj_dict)

        else:

            self.obj_dict['edges'][edge_points] = [ graph_edge.obj_dict ]


        graph_edge.set_sequence( self.get_next_sequence_number() )

        graph_edge.set_parent_graph( self.get_parent_graph() )



    def del_edge(self, src_or_list, dst=None, index=None):
        """Delete an edge from the graph.

        Given an edge's (source, destination) node names all
        matching edges(s) will be deleted if 'index' is not
        specified or set to None.
        If there are several matching edges and 'index' is
        given, only the edge in that position will be deleted.

        'index' should be an integer specifying the position
        of the edge to delete. If index is larger than the
        number of matching edges, no action is taken.

        If edges are deleted it returns True. If no action
        is taken it returns False.
        """

        if isinstance( src_or_list, (list, tuple)):
            if dst is not None and isinstance(dst, (int, long)):
                index = dst
            src, dst = src_or_list
        else:
            src, dst = src_or_list, dst

        if isinstance(src, Node):
            src = src.get_name()

        if isinstance(dst, Node):
            dst = dst.get_name()

        if self.obj_dict['edges'].has_key( (src, dst) ):

            if index is not None and index < len(self.obj_dict['edges'][(src, dst)]):
                del self.obj_dict['edges'][(src, dst)][index]
                return True
            else:
                del self.obj_dict['edges'][(src, dst)]
                return True

        return False


    def get_edge(self, src_or_list, dst=None):
        """Retrieved an edge from the graph.

        Given an edge's source and destination the corresponding
        Edge instance(s) will be returned.

        If one or more edges exist with that source and destination
        a list of Edge instances is returned.
        An empty list is returned otherwise.
        """

        if isinstance( src_or_list, (list, tuple)) and dst is None:
            edge_points = tuple(src_or_list)
            edge_points_reverse = (edge_points[1], edge_points[0])
        else:
            edge_points = (src_or_list, dst)
            edge_points_reverse = (dst, src_or_list)

        match = list()

        if self.obj_dict['edges'].has_key( edge_points ) or (
            self.get_top_graph_type() == 'graph' and self.obj_dict['edges'].has_key( edge_points_reverse )):

            edges_obj_dict = self.obj_dict['edges'].get(
                edge_points,
                self.obj_dict['edges'].get( edge_points_reverse, None ))

            for edge_obj_dict in edges_obj_dict:
                match.append( Edge( edge_points[0], edge_points[1], obj_dict = edge_obj_dict ) )

        return match


    def get_edges(self):
        return self.get_edge_list()


    def get_edge_list(self):
        """Get the list of Edge instances.

        This method returns the list of Edge instances
        composing the graph.
        """

        edge_objs = list()

        for edge, obj_dict_list in self.obj_dict['edges'].iteritems():
                edge_objs.extend( [ Edge( obj_dict = obj_d ) for obj_d in obj_dict_list ] )

        return edge_objs



    def add_subgraph(self, sgraph):
        """Adds an subgraph object to the graph.

        It takes a subgraph object as its only argument and returns
        None.
        """

        if not isinstance(sgraph, Subgraph) and not isinstance(sgraph, Cluster):
            raise TypeError('add_subgraph() received a non subgraph class object:' + str(sgraph))

        if self.obj_dict['subgraphs'].has_key(sgraph.get_name()):

            sgraph_list = self.obj_dict['subgraphs'][ sgraph.get_name() ]
            sgraph_list.append( sgraph.obj_dict )

        else:
            self.obj_dict['subgraphs'][ sgraph.get_name() ] = [ sgraph.obj_dict ]

        sgraph.set_sequence( self.get_next_sequence_number() )

        sgraph.set_parent_graph( self.get_parent_graph() )




    def get_subgraph(self, name):
        """Retrieved a subgraph from the graph.

        Given a subgraph's name the corresponding
        Subgraph instance will be returned.

        If one or more subgraphs exist with the same name, a list of
        Subgraph instances is returned.
        An empty list is returned otherwise.
        """

        match = list()

        if self.obj_dict['subgraphs'].has_key( name ):

            sgraphs_obj_dict = self.obj_dict['subgraphs'].get( name )

            for obj_dict_list in sgraphs_obj_dict:
                #match.extend( Subgraph( obj_dict = obj_d ) for obj_d in obj_dict_list )
                match.append( Subgraph( obj_dict = obj_dict_list ) )

        return match


    def get_subgraphs(self):

        return self.get_subgraph_list()


    def get_subgraph_list(self):
        """Get the list of Subgraph instances.

        This method returns the list of Subgraph instances
        in the graph.
        """

        sgraph_objs = list()

        for sgraph, obj_dict_list in self.obj_dict['subgraphs'].iteritems():
                sgraph_objs.extend( [ Subgraph( obj_dict = obj_d ) for obj_d in obj_dict_list ] )

        return sgraph_objs



    def set_parent_graph(self, parent_graph):

        self.obj_dict['parent_graph'] = parent_graph

        for obj_list in self.obj_dict['nodes'].itervalues():
            for obj in obj_list:
                obj['parent_graph'] = parent_graph

        for obj_list in self.obj_dict['edges'].itervalues():
            for obj in obj_list:
                obj['parent_graph'] = parent_graph

        for obj_list in self.obj_dict['subgraphs'].itervalues():
            for obj in obj_list:
                Graph(obj_dict=obj).set_parent_graph(parent_graph)



    def to_string(self):
        """Returns a string representation of the graph in dot language.

        It will return the graph and all its subelements in string from.
        """


        graph = list()

        if self.obj_dict.get('strict', None) is not None:

            if self==self.get_parent_graph() and self.obj_dict['strict']:

                graph.append('strict ')

        if self.obj_dict['name'] == '':
            if 'show_keyword' in self.obj_dict and self.obj_dict['show_keyword']:
                graph.append( 'subgraph {\n' )
            else:
                graph.append( '{\n' )
        else:
            graph.append( '%s %s {\n' % (self.obj_dict['type'], self.obj_dict['name']) )


        for attr in self.obj_dict['attributes'].iterkeys():

            if self.obj_dict['attributes'].get(attr, None) is not None:

                val = self.obj_dict['attributes'].get(attr)
                if val is not None:
                    graph.append( '%s=%s' % (attr, quote_if_necessary(val)) )
                else:
                    graph.append( attr )

                graph.append( ';\n' )


        edges_done = set()

        edge_obj_dicts = list()
        for e in self.obj_dict['edges'].itervalues():
            edge_obj_dicts.extend(e)

        if edge_obj_dicts:
            edge_src_set, edge_dst_set = zip( *[obj['points'] for obj in edge_obj_dicts] )
            edge_src_set, edge_dst_set = set(edge_src_set), set(edge_dst_set)
        else:
            edge_src_set, edge_dst_set = set(), set()

        node_obj_dicts = list()
        for e in self.obj_dict['nodes'].itervalues():
            node_obj_dicts.extend(e)

        sgraph_obj_dicts = list()
        for sg in self.obj_dict['subgraphs'].itervalues():
            sgraph_obj_dicts.extend(sg)


        obj_list = [ (obj['sequence'], obj) for obj in (edge_obj_dicts + node_obj_dicts + sgraph_obj_dicts) ]
        obj_list.sort()

        for idx, obj in obj_list:

            if obj['type'] == 'node':

                node = Node(obj_dict=obj)

                if self.obj_dict.get('suppress_disconnected', False):

                    if (node.get_name() not in edge_src_set and
                        node.get_name() not in edge_dst_set):

                        continue

                graph.append( node.to_string()+'\n' )

            elif obj['type'] == 'edge':

                edge = Edge(obj_dict=obj)

                if self.obj_dict.get('simplify', False) and edge in edges_done:
                    continue

                graph.append( edge.to_string() + '\n' )
                edges_done.add(edge)

            else:

                sgraph = Subgraph(obj_dict=obj)

                graph.append( sgraph.to_string()+'\n' )

        graph.append( '}\n' )

        return ''.join(graph)



class Subgraph(Graph):

    """Class representing a subgraph in Graphviz's dot language.

    This class implements the methods to work on a representation
    of a subgraph in Graphviz's dot language.

    subgraph(graph_name='subG', suppress_disconnected=False, attribute=value, ...)

    graph_name:
        the subgraph's name
    suppress_disconnected:
        defaults to false, which will remove from the
        subgraph any disconnected nodes.
    All the attributes defined in the Graphviz dot language should
    be supported.

    Attributes can be set through the dynamically generated methods:

     set_[attribute name], i.e. set_size, set_fontname

    or using the instance's attributes:

     Subgraph.obj_dict['attributes'][attribute name], i.e.

        subgraph_instance.obj_dict['attributes']['label']
        subgraph_instance.obj_dict['attributes']['fontname']
    """


    # RMF: subgraph should have all the attributes of graph so it can be passed
    # as a graph to all methods
    #
    def __init__(self, graph_name='', obj_dict=None, suppress_disconnected=False,
        simplify=False, **attrs):


        Graph.__init__(self, graph_name=graph_name, obj_dict=obj_dict,
            suppress_disconnected=suppress_disconnected, simplify=simplify, **attrs)

        if obj_dict is None:

            self.obj_dict['type'] = 'subgraph'




class Cluster(Graph):

    """Class representing a cluster in Graphviz's dot language.

    This class implements the methods to work on a representation
    of a cluster in Graphviz's dot language.

    cluster(graph_name='subG', suppress_disconnected=False, attribute=value, ...)

    graph_name:
        the cluster's name (the string 'cluster' will be always prepended)
    suppress_disconnected:
        defaults to false, which will remove from the
        cluster any disconnected nodes.
    All the attributes defined in the Graphviz dot language should
    be supported.

    Attributes can be set through the dynamically generated methods:

     set_[attribute name], i.e. set_color, set_fontname

    or using the instance's attributes:

     Cluster.obj_dict['attributes'][attribute name], i.e.

        cluster_instance.obj_dict['attributes']['label']
        cluster_instance.obj_dict['attributes']['fontname']
    """


    def __init__(self, graph_name='subG', obj_dict=None, suppress_disconnected=False,
        simplify=False, **attrs):

        Graph.__init__(self, graph_name=graph_name, obj_dict=obj_dict,
            suppress_disconnected=suppress_disconnected, simplify=simplify, **attrs)

        if obj_dict is None:

            self.obj_dict['type'] = 'subgraph'
            self.obj_dict['name'] = 'cluster_'+graph_name

        self.create_attribute_methods(CLUSTER_ATTRIBUTES)






class Dot(Graph):
    """A container for handling a dot language file.

    This class implements methods to write and process
    a dot language file. It is a derived class of
    the base class 'Graph'.
    """



    def __init__(self, *argsl, **argsd):
        Graph.__init__(self, *argsl, **argsd)

        self.shape_files = list()

        self.progs = None

        self.formats = ['canon', 'cmap', 'cmapx', 'cmapx_np', 'dia', 'dot',
            'fig', 'gd', 'gd2', 'gif', 'hpgl', 'imap', 'imap_np', 'ismap',
            'jpe', 'jpeg', 'jpg', 'mif', 'mp', 'pcl', 'pdf', 'pic', 'plain',
            'plain-ext', 'png', 'ps', 'ps2', 'svg', 'svgz', 'vml', 'vmlz',
            'vrml', 'vtx', 'wbmp', 'xdot', 'xlib' ]

        self.prog = 'dot'

        # Automatically creates all the methods enabling the creation
        # of output in any of the supported formats.
        for frmt in self.formats:
            self.__setattr__(
                'create_'+frmt,
                lambda f=frmt, prog=self.prog : self.create(format=f, prog=prog))
            f = self.__dict__['create_'+frmt]
            f.__doc__ = '''Refer to the docstring accompanying the 'create' method for more information.'''

        for frmt in self.formats+['raw']:
            self.__setattr__(
                'write_'+frmt,
                lambda path, f=frmt, prog=self.prog : self.write(path, format=f, prog=prog))

            f = self.__dict__['write_'+frmt]
            f.__doc__ = '''Refer to the docstring accompanying the 'write' method for more information.'''



    def __getstate__(self):

        dict = copy.copy(self.obj_dict)

        return dict

    def __setstate__(self, state):

        self.obj_dict = state


    def set_shape_files(self, file_paths):
        """Add the paths of the required image files.

        If the graph needs graphic objects to be used as shapes or otherwise
        those need to be in the same folder as the graph is going to be rendered
        from. Alternatively the absolute path to the files can be specified when
        including the graphics in the graph.

        The files in the location pointed to by the path(s) specified as arguments
        to this method will be copied to the same temporary location where the
        graph is going to be rendered.
        """

        if isinstance( file_paths, basestring ):
            self.shape_files.append( file_paths )

        if isinstance( file_paths, (list, tuple) ):
            self.shape_files.extend( file_paths )


    def set_prog(self, prog):
        """Sets the default program.

        Sets the default program in charge of processing
        the dot file into a graph.
        """
        self.prog = prog


    def set_graphviz_executables(self, paths):
        """This method allows to manually specify the location of the GraphViz executables.

        The argument to this method should be a dictionary where the keys are as follows:

            {'dot': '', 'twopi': '', 'neato': '', 'circo': '', 'fdp': ''}

        and the values are the paths to the corresponding executable, including the name
        of the executable itself.
        """

        self.progs = paths


    def write(self, path, prog=None, format='raw'):
        """Writes a graph to a file.

        Given a filename 'path' it will open/create and truncate
        such file and write on it a representation of the graph
        defined by the dot object and in the format specified by
        'format'.
        The format 'raw' is used to dump the string representation
        of the Dot object, without further processing.
        The output can be processed by any of graphviz tools, defined
        in 'prog', which defaults to 'dot'
        Returns True or False according to the success of the write
        operation.

        There's also the preferred possibility of using:

            write_'format'(path, prog='program')

        which are automatically defined for all the supported formats.
        [write_ps(), write_gif(), write_dia(), ...]
        """

        if prog is None:
            prog = self.prog

        dot_fd = file(path, "w+b")
        if format == 'raw':
            data = self.to_string()
            if isinstance(data, basestring):
                if not isinstance(data, unicode):
                    try:
                        data = unicode(data, 'utf-8')
                    except:
                        pass

            try:
                data = data.encode('utf-8')
            except:
                pass
            dot_fd.write(data)
        else:
            dot_fd.write(self.create(prog, format))
        dot_fd.close()

        return True



    def create(self, prog=None, format='ps'):
        """Creates and returns a Postscript representation of the graph.

        create will write the graph to a temporary dot file and process
        it with the program given by 'prog' (which defaults to 'twopi'),
        reading the Postscript output and returning it as a string is the
        operation is successful.
        On failure None is returned.

        There's also the preferred possibility of using:

            create_'format'(prog='program')

        which are automatically defined for all the supported formats.
        [create_ps(), create_gif(), create_dia(), ...]

        If 'prog' is a list instead of a string the fist item is expected
        to be the program name, followed by any optional command-line
        arguments for it:

            [ 'twopi', '-Tdot', '-s10' ]
        """

        if prog is None:
            prog = self.prog

        if isinstance(prog, (list, tuple)):
            prog, args = prog[0], prog[1:]
        else:
            args = []

        if self.progs is None:
            self.progs = find_graphviz()
            if self.progs is None:
                raise InvocationException(
                    'GraphViz\'s executables not found' )

        if not self.progs.has_key(prog):
            raise InvocationException(
                'GraphViz\'s executable "%s" not found' % prog )

        if not os.path.exists( self.progs[prog] ) or not os.path.isfile( self.progs[prog] ):
            raise InvocationException(
                'GraphViz\'s executable "%s" is not a file or doesn\'t exist' % self.progs[prog] )


        tmp_fd, tmp_name = tempfile.mkstemp()
        os.close(tmp_fd)
        self.write(tmp_name)
        tmp_dir = os.path.dirname(tmp_name )

        # For each of the image files...
        #
        for img in self.shape_files:

            # Get its data
            #
            f = file(img, 'rb')
            f_data = f.read()
            f.close()

            # And copy it under a file with the same name in the temporary directory
            #
            f = file( os.path.join( tmp_dir, os.path.basename(img) ), 'wb' )
            f.write(f_data)
            f.close()

        cmdline = [self.progs[prog], '-T'+format, tmp_name] + args

        p = subprocess.Popen(
            cmdline,
            cwd=tmp_dir,
            stderr=subprocess.PIPE, stdout=subprocess.PIPE)

        stderr = p.stderr
        stdout = p.stdout

        stdout_output = list()
        while True:
            data = stdout.read()
            if not data:
                break
            stdout_output.append(data)
        stdout.close()

        stdout_output = ''.join(stdout_output)

        if not stderr.closed:
            stderr_output = list()
            while True:
                data = stderr.read()
                if not data:
                    break
                stderr_output.append(data)
            stderr.close()

            if stderr_output:
                stderr_output = ''.join(stderr_output)

        #pid, status = os.waitpid(p.pid, 0)
        status = p.wait()

        if status != 0 :
            raise InvocationException(
                'Program terminated with status: %d. stderr follows: %s' % (
                    status, stderr_output) )
        elif stderr_output:
            print stderr_output

        # For each of the image files...
        #
        for img in self.shape_files:

            # remove it
            #
            os.unlink( os.path.join( tmp_dir, os.path.basename(img) ) )

        os.unlink(tmp_name)

        return stdout_output

