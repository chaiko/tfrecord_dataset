#! /bin/bash

# Set max line length as 100.
isort -l 100 tfrecord_dataset setup.py
yapf --style='{based_on_style: yapf, column_limit: 100}' -p -i -r tfrecord_dataset setup.py
flake8 . --ignore=F841,E124,E126,W5 --max-line-length=100 --indent-size=2 --extend-exclude=venv*
