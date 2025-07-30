from __future__ import annotations

import networkx as nx
import pytest
from swiflow import common, gflow
from swiflow.common import Plane

from tests.assets import CASES, FlowTestCase


@pytest.mark.parametrize("c", CASES)
def test_gflow(c: FlowTestCase) -> None:
    result = gflow.find(c.g, c.iset, c.oset, planes=c.planes)
    assert result == c.gflow
    if result is not None:
        gflow.verify(result, c.g, c.iset, c.oset, planes=c.planes)


def test_gflow_redundant() -> None:
    g: nx.Graph[int] = nx.Graph([(0, 1)])
    iset = {0}
    oset = {1}
    planes = {0: Plane.XY, 1: Plane.XY}
    with pytest.raises(ValueError, match=r".*Excessive measurement planes specified.*"):
        gflow.find(g, iset, oset, planes=planes)


@pytest.mark.parametrize("c", CASES)
def test_infer_verify(c: FlowTestCase) -> None:
    if c.gflow is None:
        pytest.skip()
    f, _ = c.gflow
    gflow.verify(f, c.g, c.iset, c.oset, planes=c.planes)
    layers = common.infer_layers(c.g, f)
    gflow.verify((f, layers), c.g, c.iset, c.oset, planes=c.planes)
