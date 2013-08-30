#! /usr/bin/env python
# -*- coding: utf-8 -*-


__author__ = 'Justin Bayer, bayer.justin@googlemail.com'


from setuptools import setup, find_packages


setup(
    name="breze",
    version="pre-0.1",
    description="tools for theano",
    license="BSD",
    keywords="Machine Learning Theano",
    packages=find_packages(exclude=['examples', 'docs']),
    include_package_data=True,
)

