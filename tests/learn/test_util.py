# -*- coding: utf-8 -*-

import json

import numpy as np

from breze.learn.utils import JsonForgivingEncoder


def test_json_forgiving_encoder():
    dct = {
        'bla2': np.zeros(1),
        'bla': np.zeros(3),
        'yehaw': 2,
        'x': lambda x: 2 * x,
        'y': (i for i in xrange(3)),
    }

    print json.dumps(dct, cls=JsonForgivingEncoder)
