# -*- coding: utf-8 -*-


import json
import types

import numpy as np


class KeyPrinter(object):

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, info):
        for key in self.keys:
            print '%s = %s' % (key, info.get(key, '?'))


class ForgivingEncoder(json.JSONEncoder):

    unknown_types = (
        types.FunctionType, types.GeneratorType, np.ndarray)

    def default(self, obj):
        if isinstance(obj, np.ndarray) and obj.ndim == 0:
            obj = float(obj)

        if isinstance(obj, self.unknown_types):
            return repr(obj)

        return json.JSONEncoder.default(self, obj)


class JsonPrinter(object):

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, info):
        dct = dict((k, info[k]) for k in self.keys)
        print json.dumps(dct, cls=ForgivingEncoder)


def point_print(info):
    print '.'
