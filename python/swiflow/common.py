"""Common functionalities."""

from __future__ import annotations

import itertools
from collections.abc import Hashable, Mapping
from collections.abc import Set as AbstractSet
from typing import TypeVar

import networkx as nx

from swiflow import _common
from swiflow._impl import gflow, pflow

Plane = gflow.Plane
PPlane = pflow.PPlane

_V = TypeVar("_V", bound=Hashable)

Flow = dict[_V, _V]
"""Flow map as a dictionary. :math:`f(u)` is stored in :py:obj:`f[u]`."""

GFlow = dict[_V, set[_V]]
"""Generalized flow map as a dictionary. :math:`f(u)` is stored in :py:obj:`f[u]`."""

PFlow = dict[_V, set[_V]]
"""Pauli flow map as a dictionary. :math:`f(u)` is stored in :py:obj:`f[u]`."""

Layer = dict[_V, int]
r"""Layer of each node representing the partial order. :math:`layer(u) > layer(v)` implies :math:`u \prec v`.
"""


def _infer_layer_impl(gd: nx.DiGraph[_V]) -> Mapping[_V, int]:
    pred = {u: set(gd.predecessors(u)) for u in gd.nodes}
    work = {u for u, pu in pred.items() if not pu}
    ret: dict[_V, int] = {}
    for l_now in itertools.count():
        if not work:
            break
        next_work: set[_V] = set()
        for u in work:
            ret[u] = l_now
            for v in gd.successors(u):
                ent = pred[v]
                ent.discard(u)
                if not ent:
                    next_work.add(v)
        work = next_work
    if len(ret) != len(gd):
        msg = "Failed to determine layer for all nodes."
        raise ValueError(msg)
    return ret


def infer_layer(g: nx.Graph[_V], anyflow: Mapping[_V, _V | AbstractSet[_V]]) -> Mapping[_V, int]:
    """Infer layer from flow/gflow.

    Notes
    -----
    This function is based on greedy algorithm.
    """
    gd: nx.DiGraph[_V] = nx.DiGraph()
    for u, fu_ in anyflow.items():
        fu = fu_ if isinstance(fu_, AbstractSet) else {fu_}
        fu_odd = _common.odd_neighbors(g, fu)
        for v in itertools.chain(fu, fu_odd):
            if u == v:
                continue
            gd.add_edge(u, v)
    gd = gd.reverse()
    return _infer_layer_impl(gd)
