
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
    for path, item in dfs(dl):
        dictlist.set_(dl, path, func(item))
