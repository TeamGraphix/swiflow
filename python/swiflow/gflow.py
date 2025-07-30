"""Maximally-delayed gflow algorithm.

This module provides functions to compute and verify maximally-delayed generalized flow.
See :footcite:t:`Mhalla2008` and :footcite:t:`Backens2021` for details.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from collections.abc import Set as AbstractSet
from typing import TYPE_CHECKING, TypeVar

from swiflow import _common
from swiflow._common import IndexMap
from swiflow._impl import gflow as gflow_bind
from swiflow.common import GFlow, Layer, Plane

if TYPE_CHECKING:
    import networkx as nx

_V = TypeVar("_V", bound=Hashable)
GFlowResult = tuple[GFlow[_V], Layer[_V]]


def find(
    g: nx.Graph[_V],
    iset: AbstractSet[_V],
    oset: AbstractSet[_V],
    *,
    plane: Mapping[_V, Plane] | None = None,
) -> GFlowResult[_V] | None:
    r"""Compute generalized flow.

    If it returns a gflow, it is guaranteed to be maximally-delayed, i.e., the number of layers is minimized.

    Parameters
    ----------
    g : `networkx.Graph`
        Simple graph representing MBQC pattern.
    iset : `collections.abc.Set`
        Input nodes.
    oset : `collections.abc.Set`
        Output nodes.
    plane : `collections.abc.Mapping`
        Measurement plane for each node in :math:`V \setminus O`.
        Defaults to `Plane.XY`.

    Returns
    -------
    `tuple` of gflow/layer or `None`
        Return the gflow if any, otherwise `None`.
    """
    _common.check_graph(g, iset, oset)
    vset = g.nodes
    if plane is None:
        plane = dict.fromkeys(vset - oset, Plane.XY)
    _common.check_planelike(vset, oset, plane)
    codec = IndexMap(vset)
    g_ = codec.encode_graph(g)
    iset_ = codec.encode_set(iset)
    oset_ = codec.encode_set(oset)
    plane_ = codec.encode_dictkey(plane)
    if ret_ := gflow_bind.find(g_, iset_, oset_, plane_):
        f_, layer_ = ret_
        f = codec.decode_gflow(f_)
        layer = codec.decode_layer(layer_)
        return f, layer
    return None


_GFlow = Mapping[_V, AbstractSet[_V]]
_Layer = Mapping[_V, int]


def _codec_wrap(
    codec: IndexMap[_V],
    gflow: tuple[_GFlow[_V], _Layer[_V]] | _GFlow[_V],
) -> tuple[dict[int, set[int]], list[int] | None]:
    if isinstance(gflow, tuple):
        f, layer = gflow
        return codec.encode_gflow(f), codec.encode_layer(layer)
    return codec.encode_gflow(gflow), None


def verify(
    gflow: tuple[_GFlow[_V], _Layer[_V]] | _GFlow[_V],
    g: nx.Graph[_V],
    iset: AbstractSet[_V],
    oset: AbstractSet[_V],
    *,
    plane: Mapping[_V, Plane] | None = None,
) -> None:
    r"""Verify generalized flow.

    Parameters
    ----------
    gflow : gflow (required) and layer (optional)
        Generalized flow to verify.
    g : `networkx.Graph`
        Simple graph representing MBQC pattern.
    iset : `collections.abc.Set`
        Input nodes.
    oset : `collections.abc.Set`
        Output nodes.
    plane : `collections.abc.Mapping`
        Measurement plane for each node in :math:`V \setminus O`.
        Defaults to `Plane.XY`.

    Raises
    ------
    ValueError
        If the graph is invalid or verification fails.
    """
    _common.check_graph(g, iset, oset)
    vset = g.nodes
    if plane is None:
        plane = dict.fromkeys(vset - oset, Plane.XY)
    codec = IndexMap(vset)
    g_ = codec.encode_graph(g)
    iset_ = codec.encode_set(iset)
    oset_ = codec.encode_set(oset)
    plane_ = codec.encode_dictkey(plane)
    codec.ecatch(gflow_bind.verify, _codec_wrap(codec, gflow), g_, iset_, oset_, plane_)
