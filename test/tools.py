# -*- coding: utf-8 -*-


def arr_equal(arr1, arr2, eps=1E-8):
    return (abs(arr1 - arr2) < eps).all()
