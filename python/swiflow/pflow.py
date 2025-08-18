"""Maximally-delayed Pauli flow algorithm.

This module provides functions to compute and verify maximally-delayed Pauli flow.
See :footcite:t:`Simons2021` for details.
"""

from __future__ import annotations

import warnings
from collections.abc import Hashable, Mapping
from collections.abc import Set as AbstractSet
from typing import TYPE_CHECKING, TypeVar

from swiflow import _common
from swiflow._common import IndexMap
from swiflow._impl import pflow as pflow_bind
from swiflow.common import Layers, PFlow, PPlane

if TYPE_CHECKING:
    import networkx as nx

_V = TypeVar("_V", bound=Hashable)
PFlowResult = tuple[PFlow[_V], Layers[_V]]


def find(
    g: nx.Graph[_V],
    iset: AbstractSet[_V],
    oset: AbstractSet[_V],
    *,
    pplanes: Mapping[_V, PPlane] | None = None,
) -> PFlowResult[_V] | None:
    r"""Compute Pauli flow.

    If it returns a Pauli flow, it is guaranteed to be maximally-delayed, i.e., the number of layers is minimized.

    Parameters
    ----------
    g : `networkx.Graph`
        Simple graph representing MBQC pattern.
    iset : `collections.abc.Set`
        Input nodes.
    oset : `collections.abc.Set`
        Output nodes.
    pplanes : `collections.abc.Mapping`
        Measurement plane or Pauli index for each node in :math:`V \setminus O`.
        Defaults to `PPlane.XY`.

    Returns
    -------
    `tuple` of Pauli flow/layers or `None`
        Return the Pauli flow if any, otherwise `None`.

    Notes
    -----
    Use `gflow.find` whenever possible for better performance.
    """
    _common.check_graph(g, iset, oset)
    vset = g.nodes
    if pplanes is None:
        pplanes = dict.fromkeys(vset - oset, PPlane.XY)
    _common.check_planelike(vset, oset, pplanes)
    if all(pp not in {PPlane.X, PPlane.Y, PPlane.Z} for pp in pplanes.values()):
        msg = "No Pauli measurement found. Use gflow.find instead."
        warnings.warn(msg, stacklevel=1)
    codec = IndexMap(vset)
    g_ = codec.encode_graph(g)
    iset_ = codec.encode_set(iset)
    oset_ = codec.encode_set(oset)
    pplanes_ = codec.encode_dictkey(pplanes)
    if ret_ := pflow_bind.find(g_, iset_, oset_, pplanes_):
        f_, layers_ = ret_
        f = codec.decode_gflow(f_)
        layers = codec.decode_layers(layers_)
        return f, layers
    return None


_PFlow = Mapping[_V, AbstractSet[_V]]
_Layer = Mapping[_V, int]


def _codec_wrap(
    codec: IndexMap[_V],
    pflow: tuple[_PFlow[_V], _Layer[_V]] | _PFlow[_V],
) -> tuple[dict[int, set[int]], list[int] | None]:
    if isinstance(pflow, tuple):
        f, layers = pflow
        return codec.encode_gflow(f), codec.encode_layers(layers)
    return codec.encode_gflow(pflow), None


def verify(
    pflow: tuple[_PFlow[_V], _Layer[_V]] | _PFlow[_V],
    g: nx.Graph[_V],
    iset: AbstractSet[_V],
    oset: AbstractSet[_V],
    *,
    pplanes: Mapping[_V, PPlane] | None = None,
) -> None:
    r"""Verify Pauli flow.

    Parameters
    ----------
    pflow : Pauli flow (required) and layers (optional)
        Pauli flow to verify.
    g : `networkx.Graph`
        Simple graph representing MBQC pattern.
    iset : `collections.abc.Set`
        Input nodes.
    oset : `collections.abc.Set`
        Output nodes.
    pplanes : `collections.abc.Mapping`
        Measurement plane or Pauli index for each node in :math:`V \setminus O`.
        Defaults to `PPlane.XY`.

    Raises
    ------
    ValueError
        If the graph is invalid or verification fails.
    """
    _common.check_graph(g, iset, oset)
    vset = g.nodes
    if pplanes is None:
        pplanes = dict.fromkeys(vset - oset, PPlane.XY)
    codec = IndexMap(vset)
    g_ = codec.encode_graph(g)
    iset_ = codec.encode_set(iset)
    oset_ = codec.encode_set(oset)
    pplanes_ = codec.encode_dictkey(pplanes)
    codec.ecatch(pflow_bind.verify, _codec_wrap(codec, pflow), g_, iset_, oset_, pplanes_)
