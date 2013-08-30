# -*- coding: utf-8 -*-

def roughly(a, b, eps=1E-6):
    return (abs(a - b) < eps).all()
