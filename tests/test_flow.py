from __future__ import annotations

import pytest
from swiflow import flow

from tests.assets import CASES, FlowTestCase


@pytest.mark.parametrize("c", CASES)
@pytest.mark.parametrize("opt", [True, False])
def test_flow_graphix(c: FlowTestCase, *, opt: bool) -> None:
    result = flow.find(c.g, c.iset, c.oset)
    assert result == c.flow
    if result is not None:
        flow.verify(result, c.g, c.iset, c.oset, ensure_optimal=opt)
