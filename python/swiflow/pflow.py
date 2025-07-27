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
from swiflow.common import Layer, PFlow, PPlane

if TYPE_CHECKING:
    import networkx as nx

_V = TypeVar("_V", bound=Hashable)
PFlowResult = tuple[PFlow[_V], Layer[_V]]


def find(
    g: nx.Graph[_V],
    iset: AbstractSet[_V],
    oset: AbstractSet[_V],
    *,
    pplane: Mapping[_V, PPlane] | None = None,
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
    pplane : `collections.abc.Mapping`
        Measurement plane or Pauli index for each node in :math:`V \setminus O`.
        Defaults to `PPlane.XY`.

    Returns
    -------
    `tuple` of Pauli flow/layer or `None`
        Return the Pauli flow if any, otherwise `None`.

    Notes
    -----
    Use `gflow.find` whenever possible for better performance.
    """
    _common.check_graph(g, iset, oset)
    vset = g.nodes
    if pplane is None:
        pplane = dict.fromkeys(vset - oset, PPlane.XY)
    _common.check_planelike(vset, oset, pplane)
    if all(pp not in {PPlane.X, PPlane.Y, PPlane.Z} for pp in pplane.values()):
        msg = "No Pauli measurement found. Use gflow.find instead."
        warnings.warn(msg, stacklevel=1)
    codec = IndexMap(vset)
    g_ = codec.encode_graph(g)
    iset_ = codec.encode_set(iset)
    oset_ = codec.encode_set(oset)
    pplane_ = codec.encode_dictkey(pplane)
    if ret_ := pflow_bind.find(g_, iset_, oset_, pplane_):
        f_, layer_ = ret_
        f = codec.decode_gflow(f_)
        layer = codec.decode_layer(layer_)
        return f, layer
    return None


_PFlow = Mapping[_V, AbstractSet[_V]]
_Layer = Mapping[_V, int]


def verify(
    pflow: tuple[_PFlow[_V], _Layer[_V]] | _PFlow[_V],
    g: nx.Graph[_V],
    iset: AbstractSet[_V],
    oset: AbstractSet[_V],
    *,
    pplane: Mapping[_V, PPlane] | None = None,
    ensure_optimal: bool = False,
) -> None:
    r"""Verify Pauli flow.

    Parameters
    ----------
    pflow : Pauli flow (required) and layer (optional)
        Pauli flow to verify.
        Layer is automatically computed if omitted.
    g : `networkx.Graph`
        Simple graph representing MBQC pattern.
    iset : `collections.abc.Set`
        Input nodes.
    oset : `collections.abc.Set`
        Output nodes.
    pplane : `collections.abc.Mapping`
        Measurement plane or Pauli index for each node in :math:`V \setminus O`.
        Defaults to `PPlane.XY`.
    ensure_optimal : `bool`
        Whether the pflow should be maximally-delayed. Defaults to `False`.

    Raises
    ------
    ValueError
        If the graph is invalid or verification fails.
    """
    _common.check_graph(g, iset, oset)
    if isinstance(pflow, tuple):
        f, layer = pflow
    else:
        f = pflow
        layer = _common.infer_layer(g, pflow, pplane)
    if ensure_optimal:
        _common.check_layer(layer)
    vset = g.nodes
    if pplane is None:
        pplane = dict.fromkeys(vset - oset, PPlane.XY)
    codec = IndexMap(vset)
    g_ = codec.encode_graph(g)
    iset_ = codec.encode_set(iset)
    oset_ = codec.encode_set(oset)
    pplane_ = codec.encode_dictkey(pplane)
    f_ = codec.encode_gflow(f)
    layer_ = codec.encode_layer(layer)
    codec.ecatch(pflow_bind.verify, (f_, layer_), g_, iset_, oset_, pplane_, ensure_optimal)
