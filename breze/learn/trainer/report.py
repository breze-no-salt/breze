# -*- coding: utf-8 -*-

"""Module for various reporting strategies."""

import json
import types

import numpy as np

from breze.learn.utils import JsonForgivingEncoder


class OneLinePrinter(object):
    """OneLinePrinter class.

    Attributes
    ----------

    keys : list of strings
        For each entry in this list, the corresponding key will be taken from
        the info dictionary and printed to stdout.
    """

    def __init__(self, keys, spaces=None):
        """Create OneLinePrinter object.

        Parameters
        ----------

        keys : list of strings
            For each entry in this list, the corresponding key will be taken
            from the info dictionary and printed to stdout.
        """
        self.keys = keys
        self.spaces = spaces
        self.printed_header = False

    def __call__(self, info):
        if self.spaces is None:
            if not self.printed_header:
                print '\t'.join(self.keys)
                print
                self.printed_header = True
            print '\t'.join([str(info.get(key, '?')) for key in self.keys])
        else:
            if not self.printed_header:
                self.headerformat = ''
                self.bodyformat = ''
                for key, space in zip(self.keys, self.spaces):
                    headerspace = space.partition('.')[0] if isinstance(space, basestring) else space
                    # create format cells of type '{key:<space}', i.e., a left-bound cell corresponding to key of
                    # width space
                    self.headerformat += '{{{}:>{}}} '.format(key, headerspace)
                    self.bodyformat += '{{{}:>{}}} '.format(key, space)
                print self.headerformat.format(**dict(zip(self.keys,self.keys))) # fill the headline with keys
                print
                self.printed_header = True
            print self.bodyformat.format(**info) # fill the body with content





class KeyPrinter(object):
    """KeyPrinter class.


    Attributes
    ----------

    keys : list of strings
        For each entry in this list, the corresponding key will be taken from
        the info dictionary and printed to stdout.
    """

    def __init__(self, keys):
        """Create KeyPrinter object.

        Parameters
        ----------

        keys : list of strings
            For each entry in this list, the corresponding key will be taken
            from the info dictionary and printed to stdout.
        """
        self.keys = keys

    def __call__(self, info):
        for key in self.keys:
            print '%s = %s' % (key, info.get(key, '?'))


class JsonPrinter(object):
    """JsonPrinter class.

    Prints json documents of the info dictionaries to stdout, using only the
    keys specified.


    Attributes
    ----------

    keys : list of strings
        For each entry in this list, the corresponding key will be taken from\
        the info dictionary and printed to stdout.
    """

    def __init__(self, keys):
        """Create JsonPrinter object.

        Parameters
        ----------

        keys : list of strings
            For each entry in this list, the corresponding key will be taken from
            the info dictionary and printed to stdout.
        """
        self.keys = keys

    def __call__(self, info):
        dct = dict((k, info[k]) for k in self.keys)
        print json.dumps(dct, cls=JsonForgivingEncoder)



def point_print(info):
    """Print a point to stdout."""
    print '.'
