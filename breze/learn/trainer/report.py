# -*- coding: utf-8 -*-


class KeyPrinter(object):

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, info):
        for key in self.keys:
            print '%s = %s' % (key, info.get(key, '?'))


def point_print(info):
    print '.'

