
def get(dl, path):
    item = dl
    for key in path:
        item = item[key]
    return item


def set_(dl, path, value):
    item = dl
    for key in list(path[:-1]):
        item = item[key]
    item[path[-1]] = value


def leafs(dl):
    to_visit = [()]
    visited = set()

    while True:
        if len(to_visit) == 0:
            break
        this = to_visit.pop()
        visited.add(this)

        item = get(dl, this)
        if isinstance(item, list):
            nxts = xrange(len(item))
        elif isinstance(item, dict):
            nxts = item.keys()
        else:
            yield (this, item)
            continue

        for nxt in nxts:
            nxt = tuple(list(this) + [nxt])        # Lists are not hashable.
            if nxt in visited:
                continue

            to_visit.append(nxt)


def replace(dl, func):
    for path, item in leafs(dl):
        set_(dl, path, func(item))


def copy(dl, dct_maker=dict, lst_maker=list, deep=False):
    if isinstance(dl, dict):
        cp = dct_maker()
        for key in dl:
            cp[key] = copy(dl[key], dct_maker=dct_maker, lst_maker=lst_maker,
                           deep=deep)
    elif isinstance(dl, list):
        cp = lst_maker()
        for item in dl:
            cp.append(copy(item, dct_maker=dct_maker, lst_maker=lst_maker,
                           deep=deep))
    else:
        if deep:
            cp = copy.deepcopy(dl)
        else:
            cp = dl

    return cp
