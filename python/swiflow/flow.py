"""Maximally-delayed flow algorithm.

This module provides functions to compute and verify maximally-delayed causal flow.
See :footcite:t:`Mhalla2008` for details.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import TYPE_CHECKING, TypeVar

from swiflow import _common
from swiflow._common import IndexMap
from swiflow._impl import flow as flow_bind
from swiflow.common import Flow, Layer

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet

    import networkx as nx

_V = TypeVar("_V", bound=Hashable)
FlowResult = tuple[Flow[_V], Layer[_V]]


def find(g: nx.Graph[_V], iset: AbstractSet[_V], oset: AbstractSet[_V]) -> FlowResult[_V] | None:
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
    `tuple` of flow/layer or `None`
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


_Flow = Mapping[_V, _V]
_Layer = Mapping[_V, int]


def verify(
    flow: tuple[_Flow[_V], _Layer[_V]] | _Flow[_V],
    g: nx.Graph[_V],
    iset: AbstractSet[_V],
    oset: AbstractSet[_V],
    *,
    ensure_optimal: bool = False,
) -> None:
    """Verify causal flow.

    Parameters
    ----------
    flow : flow (required) and layer (optional)
        Flow to verify.
        Layer is automatically computed if omitted.
    g : `networkx.Graph`
        Simple graph representing MBQC pattern.
    iset : `collections.abc.Set`
        Input nodes.
    oset : `collections.abc.Set`
        Output nodes.
    ensure_optimal : `bool`
        Whether the flow should be maximally-delayed. Defaults to `False`.

    Raises
    ------
    ValueError
        If the graph is invalid or verification fails.
    """
    _common.check_graph(g, iset, oset)
    f, layer = flow if isinstance(flow, tuple) else (flow, _common.infer_layers(g, flow))
    if ensure_optimal:
        _common.check_layer(layer)
    vset = g.nodes
    codec = IndexMap(vset)
    g_ = codec.encode_graph(g)
    iset_ = codec.encode_set(iset)
    oset_ = codec.encode_set(oset)
    f_ = codec.encode_flow(f)
    layer_ = codec.encode_layer(layer)
    codec.ecatch(flow_bind.verify, (f_, layer_), g_, iset_, oset_, ensure_optimal)
