"""Common functionalities."""

from __future__ import annotations

import itertools
from collections.abc import Hashable, Mapping
from collections.abc import Set as AbstractSet
from typing import TypeVar

import networkx as nx
from typing_extensions import ParamSpec

from swiflow import _common
from swiflow._impl import gflow, pflow

Plane = gflow.Plane
PPlane = pflow.PPlane

T = TypeVar("T")
V = TypeVar("V", bound=Hashable)
P = TypeVar("P", Plane, PPlane)
S = ParamSpec("S")

Flow = dict[V, V]
"""Flow map as a dictionary. :math:`f(u)` is stored in :py:obj:`f[u]`."""

GFlow = dict[V, set[V]]
"""Generalized flow map as a dictionary. :math:`f(u)` is stored in :py:obj:`f[u]`."""

PFlow = dict[V, set[V]]
"""Pauli flow map as a dictionary. :math:`f(u)` is stored in :py:obj:`f[u]`."""

Layer = dict[V, int]
r"""Layer of each node representing the partial order. :math:`layer(u) > layer(v)` implies :math:`u \prec v`.
"""


def _infer_layer_impl(gd: nx.DiGraph[V]) -> Mapping[V, int]:
    pred = {u: set(gd.predecessors(u)) for u in gd.nodes}
    work = {u for u, pu in pred.items() if not pu}
    ret: dict[V, int] = {}
    for l_now in itertools.count():
        if not work:
            break
        next_work: set[V] = set()
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


def infer_layer(g: nx.Graph[V], anyflow: Mapping[V, V | AbstractSet[V]]) -> Mapping[V, int]:
    """Infer layer from flow/gflow.

    Notes
    -----
    This function is based on greedy algorithm.
    """
    gd: nx.DiGraph[V] = nx.DiGraph()
    for u, fu_ in anyflow.items():
        fu = fu_ if isinstance(fu_, AbstractSet) else {fu_}
        fu_odd = _common.odd_neighbors(g, fu)
        for v in itertools.chain(fu, fu_odd):
            if u == v:
                continue
            gd.add_edge(u, v)
    gd = gd.reverse()
    return _infer_layer_impl(gd)
