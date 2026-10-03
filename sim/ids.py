"""Global node ids and innovation numbers (per (src, dst), for future crossover), shared by
brain and CPPN genomes. The CPPN uses its own innovation namespace."""
_innov = {}
_counters = {"innov": 0, "node": 100}


def innovation(src, dst, space=None):
    key = (src, dst) if space is None else (space, src, dst)
    if key not in _innov:
        _innov[key] = _counters["innov"]
        _counters["innov"] += 1
    return _innov[key]


def new_node_id():
    _counters["node"] += 1
    return _counters["node"]


def reserve_node_ids(max_id):
    _counters["node"] = max(_counters["node"], max_id + 1)
