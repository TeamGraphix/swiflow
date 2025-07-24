from __future__ import annotations

import networkx as nx
import pytest
from swiflow import common, pflow
from swiflow.common import PPlane

from tests.assets import CASES, FlowTestCase


@pytest.mark.filterwarnings("ignore:No Pauli measurement found")
@pytest.mark.parametrize("c", CASES)
@pytest.mark.parametrize("opt", [True, False])
def test_pflow_graphix(c: FlowTestCase, *, opt: bool) -> None:
    result = pflow.find(c.g, c.iset, c.oset, pplane=c.pplane)
    assert result == c.pflow
    if result is not None:
        pflow.verify(result, c.g, c.iset, c.oset, pplane=c.pplane, ensure_optimal=opt)


def test_pflow_nopauli() -> None:
    g: nx.Graph[int] = nx.Graph([(0, 1)])
    iset = {0}
    oset = {1}
    planes = {0: PPlane.XY}
    with pytest.warns(UserWarning, match=r".*No Pauli measurement found\. Use gflow\.find instead\..*"):
        pflow.find(g, iset, oset, pplane=planes)


def test_pflow_redundant() -> None:
    g: nx.Graph[int] = nx.Graph([(0, 1)])
    iset = {0}
    oset = {1}
    planes = {0: PPlane.X, 1: PPlane.Y}
    with pytest.raises(ValueError, match=r".*Excessive measurement planes specified.*"):
        pflow.find(g, iset, oset, pplane=planes)


@pytest.mark.parametrize("c", CASES)
def test_infer_verify(c: FlowTestCase) -> None:
    if c.pflow is None:
        pytest.skip()
    f, _ = c.pflow
    layer = common.infer_layer(c.g, f, pplane=c.pplane)
    pflow.verify((f, layer), c.g, c.iset, c.oset, pplane=c.pplane)
