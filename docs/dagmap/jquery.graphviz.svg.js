/*
 * Copyright (c) 2015 Mountainstorm
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */


 +function ($) {
  'use strict'

  // Cross Browser starts/endsWith support
  // =====================================
  String.prototype.startsWith = function(prefix) {
    return this.indexOf(prefix) == 0;
  };

  String.prototype.endsWith = function(suffix) {
    return this.indexOf(suffix, this.length - suffix.length) !== -1;
  };

  // GRAPHVIZSVG PUBLIC CLASS DEFINITION
  // ===================================

  var GraphvizSvg = function (element, options) {
    this.type       = null
    this.options    = null
    this.enabled    = null
    this.$element   = null

    this.init('graphviz.svg', element, options)
  }

  GraphvizSvg.VERSION  = '1.0.1'

  GraphvizSvg.GVPT_2_PX = 32.5 // used to ease removal of extra space

  GraphvizSvg.DEFAULTS = {
    url: null,
    svg: null,
    shrink: '0.125pt',
    tooltips: {
      init: function ($graph) {
        var $a = $(this)
        $a.tooltip({
          container: $graph,
          placement: 'auto left',
          animation: false,
          viewport: null
        }).on('hide.bs.tooltip', function() {
          // keep them visible even if you accidentally mouse over
          if ($a.attr('data-tooltip-keepvisible')) {
            return false
          }
        })
      },
      show: function () {
        var $a = $(this)
        $a.attr('data-tooltip-keepvisible', true)
        $a.tooltip('show')
      },
      hide: function () {
        var $a = $(this)
        $a.removeAttr('data-tooltip-keepvisible')
        $a.tooltip('hide')
      },
      update: function () {
        var $this = $(this)
        if ($this.attr('data-tooltip-keepvisible')) {
          $this.tooltip('show')
          return
        }
      }
    },
    zoom: true,
    highlight: {
      selected: function (col, bg) {
        return col
      },
      unselected: function (col, bg) {
        return jQuery.Color(col).transition(bg, 0.9)
      }
    },
    ready: null
  }

  GraphvizSvg.prototype.init = function (type, element, options) {
    this.enabled   = true
    this.type      = type
    this.$element  = $(element)
    this.options   = this.getOptions(options)

    if (options.url) {
      var that = this
      $.get(options.url, null, function(data) {
        var svg = $("svg", data)
        that.$element.html(document.adoptNode(svg[0]))
        that.setup()
      }, "xml")
    } else {
      if (options.svg) {
        this.$element.html(options.svg)
      }
      this.setup()
    }
  }

  GraphvizSvg.prototype.getDefaults = function () {
    return GraphvizSvg.DEFAULTS
  }

  GraphvizSvg.prototype.getOptions = function (options) {
    options = $.extend({}, this.getDefaults(), this.$element.data(), options)

    if (options.shrink) {
      if (typeof options.shrink != 'object') {
        options.shrink = {
          x: options.shrink,
          y: options.shrink
        }
      }
      options.shrink.x = this.convertToPx(options.shrink.x)
      options.shrink.y = this.convertToPx(options.shrink.y)
    }
    return options
  }

  GraphvizSvg.prototype.setup = function () {
    var options = this.options

    // save key elements in the graph for easy access
    var $svg = $(this.$element.children('svg'))
    var $graph = $svg.children('g:first')
    this.$svg = $svg
    this.$graph = $graph
    this.$background = $graph.children('polygon:first') // might not exist
    this.$nodes = $graph.children('.node')
    this.$edges = $graph.children('.edge')
    this._nodesByName = {}
    this._edgesByName = {}

    // add top level class and copy background color to element
    this.$element.addClass('graphviz-svg')
    if (this.$background.length) {
      this.$element.css('background', this.$background.attr('fill'))
    }

    // setup all the nodes and edges
    var that = this
    this.$nodes.each(function () { that.setupNodesEdges($(this), true) })
    this.$edges.each(function () { that.setupNodesEdges($(this), false) })

    // remove the graph title element
    var $title = this.$graph.children('title')
    this.$graph.attr('data-name', $title.text())
    $title.remove()

    if (options.zoom) {
      this.setupZoom()
    }

    // tell people we're done
    if (options.ready) {
      options.ready.call(this)
    }
  }

  GraphvizSvg.prototype.setupNodesEdges = function ($el, isNode) {
    var that = this
    var options = this.options

    // save the colors of the paths, ellipses and polygons
    $el.find('polygon, ellipse, path').each(function () {
      var $this = $(this)
      // save original colors
      $this.data('graphviz.svg.color', {
        fill: $this.attr('fill'),
        stroke: $this.attr('stroke')

      })

      // shrink it if it's a node
      if (isNode && options.shrink) {
        that.scaleNode($this)
      }
    })

    // save the node name and check if theres a comment above; save it
    var $title = $el.children('title')
    if ($title[0]) {
      // remove any compass points:
      var title = $title.text().replace(/:[snew][ew]?/g,'')
      $el.attr('data-name', title)
      $title.remove()
      if (isNode) {
        this._nodesByName[title] = $el[0]
      } else {
        this._edgesByName[title] = $el[0]
      }
      // without a title we can't tell if its a user comment or not
      var previousSibling = $el[0].previousSibling
      while (previousSibling && previousSibling.nodeType != 8) {
        previousSibling = previousSibling.previousSibling
      }
      if (previousSibling != null && previousSibling.nodeType == 8) {
        var htmlDecode = function (input) {
          var e = document.createElement('div')
          e.innerHTML = input
          return e.childNodes[0].nodeValue
        }
        var value = htmlDecode(previousSibling.nodeValue.trim())
        if (value != title) {
          // user added comment
          $el.attr('data-comment', value)
        }
      }
    }

    // remove namespace from a[xlink:title]
    $el.find('a').filter(function () { 
      return $(this).attr('xlink:title') }).each(function () {
        var $a = $(this)
        $a.attr('title', $a.attr('xlink:title'))
        $a.removeAttr('xlink:title')
        if (options.tooltips) {
          options.tooltips.init.call(this, that.$element)
        }
      })
  }

  GraphvizSvg.prototype.setupZoom = function() {
    var that = this
    var $element = this.$element
    var $svg = this.$svg
    this.zoom = {width: $svg.attr('width'), height: $svg.attr('height'), percentage: null }
    this.scaleView(100.0)
    $element.mousewheel(function (evt) {
        if (evt.shiftKey) {
          var percentage = that.zoom.percentage
          percentage -= evt.deltaY * evt.deltaFactor
          if (percentage < 100.0) {
            percentage = 100.0
          }
          // get pointer offset in view
          // ratio offset within svg
          var dx = evt.pageX - $svg.offset().left
          var dy = evt.pageY - $svg.offset().top
          var rx = dx / $svg.width()
          var ry = dy / $svg.height()

          // offset within frame ($element)
          var px = evt.pageX - $element.offset().left
          var py = evt.pageY - $element.offset().top

          that.scaleView(percentage)
          // scroll so pointer is still in same place
          $element.scrollLeft((rx * $svg.width()) + 0.5 - px)
          $element.scrollTop((ry * $svg.height()) + 0.5 - py)
          return false // stop propagation
        }
      })
  }

  GraphvizSvg.prototype.scaleView = function(percentage) {
    var that = this
    var $svg = this.$svg
    $svg.attr('width', percentage + '%')
    $svg.attr('height', percentage + '%')
    this.zoom.percentage = percentage
    // now callback to update tooltip position
    var $everything = this.$nodes.add(this.$edges)
    $everything.children('a[title]').each(function () {
      that.options.tooltips.update.call(this)
    })
  }

  GraphvizSvg.prototype.scaleNode = function($node) {
    var dx = this.options.shrink.x
    var dy = this.options.shrink.y
    var tagName = $node.prop('tagName')
    if (tagName == 'ellipse') {
      $node.attr('rx', parseFloat($node.attr('rx')) - dx)
      $node.attr('ry', parseFloat($node.attr('ry')) - dy)
    } else if (tagName == 'polygon') {
      // this is more complex - we need to scale it manually
      var bbox = $node[0].getBBox()
      var cx = bbox.x + (bbox.width / 2)
      var cy = bbox.y + (bbox.height / 2)
      var pts = $node.attr('points').split(' ')
      var points = '' // new value
      for (var i in pts) {
        var xy = pts[i].split(',')
        var ox = parseFloat(xy[0])
        var oy = parseFloat(xy[1])
        points += (((cx - ox) / (bbox.width / 2) * dx) + ox) +
          ',' +
          (((cy - oy) / (bbox.height / 2) * dy) + oy) +
          ' '
      }
      $node.attr('points', points)
    }
  }

  GraphvizSvg.prototype.convertToPx = function (val) {
    var retval = val
    if (typeof val == 'string') {
      var end = val.length
      var factor = 1.0
      if (val.endsWith('px')) {
        end -= 2
      } else if (val.endsWith('pt')) {
        end -= 2
        factor = GraphvizSvg.GVPT_2_PX
      }
      retval = parseFloat(val.substring(0, end)) * factor
    }
    return retval
  }

  GraphvizSvg.prototype.findEdge = function (nodeName, testEdge, $retval) {
    var retval = []
    for (var name in this._edgesByName) {
      var match = testEdge(nodeName, name)
      if (match) {
        if ($retval) {
          $retval.push(this._edgesByName[name])
        }
        retval.push(match)
      }
    }
    return retval
  }

  GraphvizSvg.prototype.findLinked = function (node, includeEdges, testEdge, $retval) {
    var that = this
    var $node = $(node)
    var $edges = null
    if (includeEdges) {
      $edges = $retval
    }
    var names = this.findEdge($node.attr('data-name'), testEdge, $edges)
    for (var i in names) {
      var n = this._nodesByName[names[i]]
      if (!$retval.is(n)) {
        $retval.push(n)
        that.findLinked(n, includeEdges, testEdge, $retval)
      }
    }
  }

  GraphvizSvg.prototype.colorElement = function ($el, getColor) {
    var bg = this.$element.css('background')
    $el.find('polygon, ellipse, path').each(function() {
      var $this = $(this)
      var color = $this.data('graphviz.svg.color')
      if (color.fill && $this.prop('tagName') != 'path') {
        $this.attr('fill', getColor(color.fill, bg)) // don't set  fill if it's a path
      }
      if (color.stroke) {
        $this.attr('stroke', getColor(color.stroke, bg))
      }
    })
  }

  GraphvizSvg.prototype.restoreElement = function ($el) {
    $el.find('polygon, ellipse, path').each(function() {
      var $this = $(this)
      var color = $this.data('graphviz.svg.color')
      if (color.fill) {
        $this.attr('fill', color.fill) // don't set  fill if it's a path
      }
      if (color.stroke) {
        $this.attr('stroke', color.stroke)
      }
    })
  }


  // methods users can actually call
  GraphvizSvg.prototype.nodes = function () {
    return this.$nodes
  }

  GraphvizSvg.prototype.edges = function () {
    return this.$edges
  }

  GraphvizSvg.prototype.nodesByName = function () {
    return this._nodesByName
  }

  GraphvizSvg.prototype.edgesByName = function () {
    return this._edgesByName
  }

  GraphvizSvg.prototype.linkedTo = function (node, includeEdges) {
    var $retval = $()
    this.findLinked(node, includeEdges, function (nodeName, edgeName) {
      var other = null;
      var match = '->' + nodeName
      if (edgeName.endsWith(match)) {
        other = edgeName.substring(0, edgeName.length - match.length);
      }
      return other;
    }, $retval)
    return $retval
  }

  GraphvizSvg.prototype.linkedFrom = function (node, includeEdges) {
    var $retval = $()
    this.findLinked(node, includeEdges, function (nodeName, edgeName) {
      var other = null;
      var match = nodeName + '->'
      if (edgeName.startsWith(match)) {
        other = edgeName.substring(match.length);
      }
      return other;
    }, $retval)
    return $retval
  }

  GraphvizSvg.prototype.linked = function (node, includeEdges) {
    var $retval = $()
    this.findLinked(node, includeEdges, function (nodeName, edgeName) {
      return '^' + name + '--(.*)$'
    }, $retval)
    this.findLinked(node, includeEdges, function (nodeName, edgeName) {
      return '^(.*)--' + name + '$'
    }, $retval)
    return $retval
  }

  GraphvizSvg.prototype.tooltip = function ($elements, show) {
    var that = this
    var options = this.options
    $elements.each(function () {
      $(this).find('a[title]').each(function () {
        if (show) {
          options.tooltips.show.call(this)
        } else {
          options.tooltips.hide.call(this)
        }
      })
    })
  }

  GraphvizSvg.prototype.bringToFront = function ($elements) {
    $elements.detach().appendTo(this.$graph)
  }

  GraphvizSvg.prototype.sendToBack = function ($elements) {
    if (this.$background.length) {
      $element.insertAfter(this.$background)
    } else {
      $elements.detach().prependTo(this.$graph)
    }
  }

  GraphvizSvg.prototype.highlight = function ($nodesEdges, tooltips) {
    var that = this
    var options = this.options
    var $everything = this.$nodes.add(this.$edges)
    if ($nodesEdges && $nodesEdges.length > 0) {
      // create set of all other elements and dim them
      $everything.not($nodesEdges).each(function () {
        that.colorElement($(this), options.highlight.unselected)
        $(this).css('font-weight', 'normal')
        that.tooltip($(this))
      })
      $nodesEdges.each(function () {
        that.colorElement($(this), options.highlight.selected)
        $(this).css('font-weight', 'normal')
      })

      this.tooltip($nodesEdges, tooltips)
    } else {
      $everything.each(function () {
        that.restoreElement($(this))
        $(this).css('font-weight', 'normal')
      })
      this.tooltip($everything)
    }
  }

  GraphvizSvg.prototype.destroy = function () {
    var that = this
    this.hide(function () {
      that.$element.off('.' + that.type).removeData(that.type)
    })
  }


  // GRAPHVIZSVG PLUGIN DEFINITION
  // =============================

  function Plugin(option) {
    return this.each(function () {
      var $this   = $(this)
      var data    = $this.data('graphviz.svg')
      var options = typeof option == 'object' && option

      if (!data && /destroy/.test(option)) return
      if (!data) $this.data('graphviz.svg', (data = new GraphvizSvg(this, options)))
      if (typeof option == 'string') data[option]()
    })
  }

  var old = $.fn.graphviz

  $.fn.graphviz             = Plugin
  $.fn.graphviz.Constructor = GraphvizSvg


  // GRAPHVIZ NO CONFLICT
  // ====================

  $.fn.graphviz.noConflict = function () {
    $.fn.graphviz = old
    return this
  }

}(jQuery)
