=========================================================
Using the No-Python Rewrite Pass for Fun and Optimization
=========================================================

Overview
========

This section introduces IR rewrites, and how they can be used to
implement optimizations.


Rewriting Passes
================

Rewriting passes have a simple match/apply interface.


The Rewrite Registry
====================

When you want to include a rewrite in the rewrite pass, you should
register it with the rewrite registry.


Case study: Array Expressions
=============================

This section looks at the array expression rewriter in more depth.


Conclusions and Caveats
=======================

This section reviews rewrites, and provides guidance for possible
stumbling blocks when using them.
