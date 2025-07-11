"""Maximally-delayed flow algorithm.

This module provides functions to compute and verify maximally-delayed causal flow.
See :footcite:t:`Mhalla2008` for details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from swiflow import _common
from swiflow._common import IndexMap
from swiflow._impl import flow as flow_bind
from swiflow.common import Flow, Layer, V

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet

    import networkx as nx

FlowResult = tuple[Flow[V], Layer[V]]


def find(g: nx.Graph[V], iset: AbstractSet[V], oset: AbstractSet[V]) -> FlowResult[V] | None:
    """Compute causal flow.

    If it returns a flow, it is guaranteed to be maximally-delayed, i.e., the number of layers is minimized.

    Parameters
    ----------
    g : `networkx.Graph`
        Simple graph representing MBQC pattern.
    iset : `collections.abc.Set`
        Input nodes.
    oset : `collections.abc.Set`
        Output nodes.

    Returns
    -------
    `FlowResult` or `None`
        Return the flow if any, otherwise `None`.
    """
    _common.check_graph(g, iset, oset)
    vset = g.nodes
    codec = IndexMap(vset)
    g_ = codec.encode_graph(g)
    iset_ = codec.encode_set(iset)
    oset_ = codec.encode_set(oset)
    if ret_ := flow_bind.find(g_, iset_, oset_):
        f_, layer_ = ret_
        f = codec.decode_flow(f_)
        layer = codec.decode_layer(layer_)
        return f, layer
    return None


def verify(
    flow: FlowResult[V],
    g: nx.Graph[V],
    iset: AbstractSet[V],
    oset: AbstractSet[V],
    *,
    ensure_optimal: bool = True,
) -> None:
    """Verify causal flow.

    Parameters
    ----------
    flow : `FlowResult`
        Flow to verify.
    g : `networkx.Graph`
        Simple graph representing MBQC pattern.
    iset : `collections.abc.Set`
        Input nodes.
    oset : `collections.abc.Set`
        Output nodes.
    ensure_optimal : `bool`
        Wether the flow should be maximally-delayed. Defaults to `True`.

    Raises
    ------
    ValueError
        If the graph is invalid or verification fails.
    """
    _common.check_graph(g, iset, oset)
    vset = g.nodes
    codec = IndexMap(vset)
    g_ = codec.encode_graph(g)
    iset_ = codec.encode_set(iset)
    oset_ = codec.encode_set(oset)
    f, layer = flow
    f_ = codec.encode_flow(f)
    layer_ = codec.encode_layer(layer)
    codec.ecatch(flow_bind.verify, (f_, layer_), g_, iset_, oset_, ensure_optimal)
