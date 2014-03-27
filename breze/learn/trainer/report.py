# -*- coding: utf-8 -*-


import json


class KeyPrinter(object):

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, info):
        for key in self.keys:
            print '%s = %s' % (key, info.get(key, '?'))


class JsonPrinter(object):

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, info):
        dct = dict((k, info[k]) for k in self.keys)
        print json.dumps(dct)


def point_print(info):
    print '.'
