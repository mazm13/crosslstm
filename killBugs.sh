#!/bin/bash
for i in `ls *.py`; do pylint --errors-only $i; done
