# DAG Roadmap

This directory includes a representation of the Numba roadmap in the form of a
DAG.  We have done this to enable a highly granular display of enhancements to
Numba that also shows the relationships between these tasks. Many tasks have
prerequisites, and we've found that issue trackers, Kanban boards, and
time-bucketed roadmap documentation all fail to represent this information in
different ways.

## Requirements

```
conda install jinja2 python-graphviz pyyaml
```

## Usage

```
./render.py -o dagmap.html dagmap.yaml
```

The generated HTML file will look for `jquery.graphviz.svg.js` in the same
directory.

## Updating the DAG

Copy one of the existing tasks and edit:
  * `label`: text appears on the node.  Embed `\n` for line breaks.
  * `id`: Referenced to indicate a dependency
  * `description`: Shown in the tooltip.  Automatically word-wrapped.
  * `depends_on`: Optional list of task IDs which this task depends on.

The `style` section of the file is not used yet.

## Notes

The HTML rendering of the graph is based on a slightly modified version of
(jquery.graphviz.svg)[https://github.com/mountainstorm/jquery.graphviz.svg/].
Its license is:
```
Copyright (c) 2015 Mountainstorm
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```