"""Common functionalities."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TypeVar

from typing_extensions import ParamSpec

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
