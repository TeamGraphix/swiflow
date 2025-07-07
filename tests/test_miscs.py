from __future__ import annotations

import subprocess  # noqa: S404
import sys
from pathlib import Path

import pytest
import tomli

SRC_EXAMPLES = list(Path("examples").glob("*.py"))
assert SRC_EXAMPLES


@pytest.mark.parametrize("f", SRC_EXAMPLES)
def test_examples(f: Path) -> None:
    # MEMO: Possibly insecure!
    subprocess.run(  # noqa: S603
        [sys.executable, str(f)], check=True
    )


def test_versions() -> None:
    with Path("pyproject.toml").open("rb") as f:
        v_py = tomli.load(f)["project"]["version"]

    with Path("Cargo.toml").open("rb") as f:
        v_rs = tomli.load(f)["package"]["version"]

    assert isinstance(v_py, str)
    assert isinstance(v_rs, str)
    assert v_py == v_rs
