"""Common functionalities."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TypeVar

from swiflow._impl import gflow, pflow

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
