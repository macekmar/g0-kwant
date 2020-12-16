#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='g0Kwant',
      packages=find_packages(exclude=['tests*']),
      zip_safe=False
    )
