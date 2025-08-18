class PPlane:
    XY: PPlane
    YZ: PPlane
    XZ: PPlane
    X: PPlane
    Y: PPlane
    Z: PPlane

def find(
    g: list[set[int]], iset: set[int], oset: set[int], pplanes: dict[int, PPlane]
) -> tuple[dict[int, set[int]], list[int]] | None: ...
def verify(
    pflow: tuple[dict[int, set[int]], list[int] | None],
    g: list[set[int]],
    iset: set[int],
    oset: set[int],
    pplanes: dict[int, PPlane],
) -> None: ...
