from __future__ import annotations

import networkx as nx
import pytest
from swiflow import gflow
from swiflow.common import Plane

from tests.assets import CASES, FlowTestCase


@pytest.mark.parametrize("c", CASES)
def test_gflow_graphix(c: FlowTestCase) -> None:
    result = gflow.find(c.g, c.iset, c.oset, c.plane)
    assert result == c.gflow
    if result is not None:
        gflow.verify(result, c.g, c.iset, c.oset, c.plane)


def test_gflow_redundant() -> None:
    g: nx.Graph[int] = nx.Graph([(0, 1)])
    iset = {0}
    oset = {1}
    planes = {0: Plane.XY, 1: Plane.XY}
    with pytest.raises(ValueError, match=r".*Excessive measurement planes specified.*"):
        gflow.find(g, iset, oset, planes)
