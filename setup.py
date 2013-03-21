#! /usr/bin/env python
# -*- coding: utf-8 -*-


__authors__ = ['Justin Bayer, bayer.justin@googlemail.com',
               'Sebastian Urban, surban@tum.de']


from setuptools import setup, find_packages


setup(
    name="brummlearn",
    version="pre-0.1",
    description="machine learning",
    keywords="Machine Learning",
    packages=find_packages(exclude=['examples', 'docs']),
    include_package_data=True,
)

