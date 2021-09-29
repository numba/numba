#!/bin/bash

set -v -e

which gdb
file $(which gdb)
gdb --version
