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


def _codec_wrap(
    codec: IndexMap[_V],
    flow: tuple[_Flow[_V], _Layer[_V]] | _Flow[_V],
) -> tuple[dict[int, int], list[int] | None]:
    if isinstance(flow, tuple):
        f, layer = flow
        return codec.encode_flow(f), codec.encode_layer(layer)
    return codec.encode_flow(flow), None


def verify(
    flow: tuple[_Flow[_V], _Layer[_V]] | _Flow[_V],
    g: nx.Graph[_V],
    iset: AbstractSet[_V],
    oset: AbstractSet[_V],
) -> None:
    """Verify causal flow.

    Parameters
    ----------
    flow : flow (required) and layer (optional)
        Flow to verify.
    g : `networkx.Graph`
        Simple graph representing MBQC pattern.
    iset : `collections.abc.Set`
        Input nodes.
    oset : `collections.abc.Set`
        Output nodes.

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
    codec.ecatch(flow_bind.verify, _codec_wrap(codec, flow), g_, iset_, oset_)
