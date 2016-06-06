#!/bin/bash

$PYTHON buildscripts/remove_unwanted_files.py
$PYTHON setup.py build install
