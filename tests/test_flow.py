from __future__ import annotations

import pytest
from swiflow import common, flow

from tests.assets import CASES, FlowTestCase


@pytest.mark.parametrize("c", CASES)
def test_flow(c: FlowTestCase) -> None:
    result = flow.find(c.g, c.iset, c.oset)
    assert result == c.flow
    if result is not None:
        flow.verify(result, c.g, c.iset, c.oset)


@pytest.mark.parametrize("c", CASES)
def test_infer_verify(c: FlowTestCase) -> None:
    if c.flow is None:
        pytest.skip()
    f, _ = c.flow
    flow.verify(f, c.g, c.iset, c.oset)
    layer = common.infer_layers(c.g, f)
    flow.verify((f, layer), c.g, c.iset, c.oset)
