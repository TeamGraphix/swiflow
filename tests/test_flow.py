from __future__ import annotations

import pytest
from swiflow import _common, flow

from tests.assets import CASES, FlowTestCase


@pytest.mark.parametrize("c", CASES)
@pytest.mark.parametrize("opt", [True, False])
def test_flow(c: FlowTestCase, *, opt: bool) -> None:
    result = flow.find(c.g, c.iset, c.oset)
    assert result == c.flow
    if result is not None:
        flow.verify(result, c.g, c.iset, c.oset, ensure_optimal=opt)


@pytest.mark.parametrize("c", CASES)
def test_infer_verify(c: FlowTestCase) -> None:
    if c.flow is None:
        pytest.skip()
    f, _ = c.flow
    layer = _common.infer_layer(c.g, f)
    flow.verify((f, layer), c.g, c.iset, c.oset)
