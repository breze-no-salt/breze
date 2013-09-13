#! /usr/bin/env python
# -*- coding: utf-8 -*-


__author__ = 'Justin Bayer, bayer.justin@googlemail.com'


from setuptools import setup, find_packages


setup(
    name="breze",
    version="pre-0.1",
    description="tools for machine learning via theano and others",
    license="BSD",
    keywords="Machine Learning, Theano, Deep Learning",
    packages=find_packages(exclude=['examples', 'docs']),
    include_package_data=True,
)

