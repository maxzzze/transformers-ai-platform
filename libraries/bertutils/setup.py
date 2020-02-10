#!/usr/bin/env python

import io
import os

from setuptools import find_packages
from setuptools import setup

import subprocess
import sys

def read_file(file_name):
    """
    File read wrapper for loading data unmodified from arbitrary file.
    """
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    with open(file_path, "r") as fin:
        return [line for line in fin if not line.startswith(("#", "--"))]


config = dict(
    name='bertutils',
    packages=find_packages('src'),
    install_requires=read_file("requirements.txt"),
    package_dir={'': 'src'},
    zip_safe=False,
    setup_requires=[
        'setuptools_scm',
        'setuptools-lint',
        'wheel'
    ],
    entry_points={
        'console_scripts': [
            'format-csvs-bert = bertutils.dformatter:format_for_bert_main'
        ]
    }
)

setup(**config)