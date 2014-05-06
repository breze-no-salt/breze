# -*- coding: utf-8 -*-

from breze.utils import dictlist
from attrdict import AttrDict


def make_dictlist():
    return {
        'bla': 2,
        'blo': [(2, 3), 5],
        'blu': {'foo': (2, 4),
                'far': [1, 2]},
        'blubb': (2, 3),
    }


def test_dictlist_dfs():
    tree = make_dictlist()
    contents = sorted(list(dictlist.leafs(tree)))

    assert contents[0] == (('bla',), 2)
    assert contents[1] == (('blo', 0), (2, 3))
    assert contents[2] == (('blo', 1), 5)
    assert contents[3] == (('blu', 'far', 0), 1)
    assert contents[4] == (('blu', 'far', 1), 2)
    assert contents[5] == (('blu', 'foo'), (2, 4))
    assert contents[6] == (('blubb',), (2, 3))


def test_dictlist_get():
    tree = make_dictlist()
    assert dictlist.get(tree, ('blu', 'far', 0)) == 1
    assert dictlist.get(tree, ('blu', 'foo')) == (2, 4)


def test_copy():
    tree = make_dictlist()
    tree2 = dictlist.copy(tree, dct_maker=AttrDict)

    print tree2
