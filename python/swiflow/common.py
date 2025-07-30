"""Common functionalities."""

from __future__ import annotations

import itertools
from collections.abc import Hashable, Mapping, MutableSet
from collections.abc import Set as AbstractSet
from typing import TYPE_CHECKING, TypeVar

from swiflow import _common
from swiflow._impl import gflow, pflow

if TYPE_CHECKING:
    import networkx as nx

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


def _infer_layers_impl(pred: Mapping[_V, MutableSet[_V]], succ: Mapping[_V, AbstractSet[_V]]) -> Mapping[_V, int]:
    """Fix flow layers one by one depending on order constraints.

    Notes
    -----
    :py:obj:`pred` is mutated in-place.
    """
    work = {u for u, pu in pred.items() if not pu}
    ret: dict[_V, int] = {}
    for l_now in itertools.count():
        if not work:
            break
        next_work: set[_V] = set()
        for u in work:
            ret[u] = l_now
            for v in succ[u]:
                ent = pred[v]
                ent.discard(u)
                if not ent:
                    next_work.add(v)
        work = next_work
    if len(ret) != len(succ):
        msg = "Failed to determine layer for all nodes."
        raise ValueError(msg)
    return ret


def _is_special(
    pp: PPlane | None,
    in_fu: bool,  # noqa: FBT001
    in_fu_odd: bool,  # noqa: FBT001
) -> bool:
    if pp == PPlane.X:
        return in_fu
    if pp == PPlane.Y:
        return in_fu and in_fu_odd
    if pp == PPlane.Z:
        return in_fu_odd
    return False


def _special_edges(
    g: nx.Graph[_V],
    anyflow: Mapping[_V, _V | AbstractSet[_V]],
    pplane: Mapping[_V, PPlane] | None,
) -> set[tuple[_V, _V]]:
    """Compute special edges that can bypass partial order constraints in Pauli flow."""
    ret: set[tuple[_V, _V]] = set()
    if pplane is None:
        return ret
    for u, fu_ in anyflow.items():
        fu = fu_ if isinstance(fu_, AbstractSet) else {fu_}
        fu_odd = _common.odd_neighbors(g, fu)
        for v in itertools.chain(fu, fu_odd):
            if u == v:
                continue
            if _is_special(pplane.get(v), v in fu, v in fu_odd):
                ret.add((u, v))
    return ret


def infer_layers(
    g: nx.Graph[_V],
    anyflow: Mapping[_V, _V | AbstractSet[_V]],
    pplane: Mapping[_V, PPlane] | None = None,
) -> Mapping[_V, int]:
    """Infer layer from flow/gflow using greedy algorithm.

    Parameters
    ----------
    g : `networkx.Graph`
        Simple graph representing MBQC pattern.
    anyflow : `tuple` of flow-like/layer
        Flow to verify. Compatible with both flow and generalized flow.
    pplane : `collections.abc.Mapping`, optional
        Measurement plane or Pauli index.

    Notes
    -----
    This function operates in Pauli flow mode only when :py:obj`pplane` is explicitly given.
    """
    special = _special_edges(g, anyflow, pplane)
    pred: dict[_V, set[_V]] = {u: set() for u in g.nodes}
    succ: dict[_V, set[_V]] = {u: set() for u in g.nodes}
    for u, fu_ in anyflow.items():
        fu = fu_ if isinstance(fu_, AbstractSet) else {fu_}
        fu_odd = _common.odd_neighbors(g, fu)
        for v in itertools.chain(fu, fu_odd):
            if u == v or (u, v) in special:
                continue
            # Reversed
            pred[u].add(v)
            succ[v].add(u)
    # MEMO: `pred` is invalidated by `_infer_layers_impl`
    return _infer_layers_impl(pred, succ)
