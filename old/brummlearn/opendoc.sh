#!/bin/sh
set -e

cd docs
make html 
xdg-open build/html/index.html
