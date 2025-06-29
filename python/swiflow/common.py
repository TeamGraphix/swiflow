"""Common functionalities."""

from __future__ import annotations

import dataclasses
from collections.abc import Hashable
from typing import Generic, TypeVar

from typing_extensions import ParamSpec

from swiflow._impl import gflow, pflow

Plane = gflow.Plane
PPlane = pflow.PPlane

T = TypeVar("T")
V = TypeVar("V", bound=Hashable)
P = TypeVar("P", Plane, PPlane)
S = ParamSpec("S")


@dataclasses.dataclass(frozen=True)
class FlowResult(Generic[V]):
    r"""Causal flow of an open graph."""

    f: dict[V, V]
    """Flow map as a dictionary. :math:`f(u)` is stored in :py:obj:`f[u]`."""
    layer: dict[V, int]
    r"""Layer of each node representing the partial order. :math:`layer(u) > layer(v)` implies :math:`u \prec v`.
    """


@dataclasses.dataclass(frozen=True)
class GFlowResult(Generic[V]):
    r"""Generalized flow of an open graph."""

    f: dict[V, set[V]]
    """Generalized flow map as a dictionary. :math:`f(u)` is stored in :py:obj:`f[u]`."""
    layer: dict[V, int]
    r"""Layer of each node representing the partial order. :math:`layer(u) > layer(v)` implies :math:`u \prec v`.
    """
