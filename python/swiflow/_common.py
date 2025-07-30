"""Private common functionalities."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping
from collections.abc import Set as AbstractSet
from typing import Generic, TypeVar

import networkx as nx
from typing_extensions import ParamSpec

from swiflow._impl import FlowValidationMessage

_V = TypeVar("_V", bound=Hashable)


def check_graph(g: nx.Graph[_V], iset: AbstractSet[_V], oset: AbstractSet[_V]) -> None:
    """Check if `(g, iset, oset)` is a valid open graph for MBQC.

    Raises
    ------
    TypeError
        If input types are incorrect.
    ValueError
        If the graph is empty, not simple, or `iset`/`oset` is not a subset of nodes.
    """
    if not isinstance(g, nx.Graph):
        msg = "g must be a networkx.Graph."
        raise TypeError(msg)
    if not isinstance(iset, AbstractSet):
        msg = "iset must be a set."
        raise TypeError(msg)
    if not isinstance(oset, AbstractSet):
        msg = "oset must be a set."
        raise TypeError(msg)
    if len(g) == 0:
        msg = "Graph is empty."
        raise ValueError(msg)
    if any(True for _ in nx.selfloop_edges(g)):
        msg = "Self-loop detected."
        raise ValueError(msg)
    vset = set(g.nodes)
    if not (iset <= vset):
        msg = "iset must be a subset of the nodes."
        raise ValueError(msg)
    if not (oset <= vset):
        msg = "oset must be a subset of the nodes."
        raise ValueError(msg)


def check_planelike(vset: AbstractSet[_V], oset: AbstractSet[_V], plike: Mapping[_V, _P]) -> None:
    r"""Check if measurement description is valid.

    Parameters
    ----------
    vset : `collections.abc.Set`
        All nodes.
    oset : `collections.abc.Set`
        Output nodes.
    plike : `collections.abc.Mapping`
        Measurement plane or Pauli index for each node in :math:`V \setminus O`.

    Raises
    ------
    TypeError
        If input types are incorrect.
    ValueError
        If `plike` is not a subset of `vset`, or `plike` does not cover all :math:`V \setminus O`.
    """
    if not isinstance(plike, Mapping):
        msg = "Measurement planes must be passed as a mapping."
        raise TypeError(msg)
    if not (plike.keys() <= vset):
        msg = "Cannot find corresponding nodes in the graph."
        raise ValueError(msg)
    ocset = vset - oset
    if not (ocset <= plike.keys()):
        msg = "Measurement planes should be specified for all u in V\\O."
        raise ValueError(msg)
    if not (plike.keys() <= ocset):
        msg = "Excessive measurement planes specified."
        raise ValueError(msg)


def odd_neighbors(g: nx.Graph[_V], kset: AbstractSet[_V]) -> set[_V]:
    """Compute odd neighbors of `kset` in `g`."""
    ret: set[_V] = set()
    for k in kset:
        ret.symmetric_difference_update(g.neighbors(k))
    return ret


_T = TypeVar("_T")
_P = TypeVar("_P")
_S = ParamSpec("_S")


class IndexMap(Generic[_V]):
    """Map between `V` and 0-based indices."""

    __v2i: dict[_V, int]
    __i2v: list[_V]

    def __init__(self, vset: AbstractSet[_V]) -> None:
        """Initialize the map from `vset`.

        Parameters
        ----------
        vset : `collections.abc.Set`
            Set of nodes.
            Can be any hashable type.

        Notes
        -----
        If `vset` is ordered, the indices will be assigned in the sorted order.
        """
        try:
            self.__i2v = sorted(vset)  # type: ignore[type-var]
        except Exception:  # noqa: BLE001
            self.__i2v = list(vset)
        self.__v2i = {v: i for i, v in enumerate(self.__i2v)}

    def encode(self, v: _V) -> int:
        """Encode `v` to the index.

        Returns
        -------
        `int`
            Index of `v`.

        Raises
        ------
        ValueError
            If `v` is not initially registered.
        """
        ind = self.__v2i.get(v)
        if ind is None:
            msg = f"{v} not found."
            raise ValueError(msg)
        return ind

    def encode_graph(self, g: nx.Graph[_V]) -> list[set[int]]:
        """Encode graph.

        Returns
        -------
        `g` with transformed nodes.
        """
        return [self.encode_set(g[v].keys()) for v in self.__i2v]

    def encode_set(self, vset: AbstractSet[_V]) -> set[int]:
        """Encode set."""
        return {self.encode(v) for v in vset}

    def encode_dictkey(self, mapping: Mapping[_V, _P]) -> dict[int, _P]:
        """Encode dict key.

        Returns
        -------
        `mapping` with transformed keys.
        """
        return {self.encode(k): v for k, v in mapping.items()}

    def encode_flow(self, f: Mapping[_V, _V]) -> dict[int, int]:
        """Encode flow.

        Returns
        -------
        `f` with both keys and values transformed.
        """
        return {self.encode(i): self.encode(j) for i, j in f.items()}

    def encode_gflow(self, f: Mapping[_V, AbstractSet[_V]]) -> dict[int, set[int]]:
        """Encode gflow.

        Returns
        -------
        `f` with both keys and values transformed.
        """
        return {self.encode(i): self.encode_set(si) for i, si in f.items()}

    def encode_layer(self, layer: Mapping[_V, int]) -> list[int]:
        """Encode layer.

        Returns
        -------
        `layer` values transformed.

        Notes
        -----
        `list` is used instead of `dict` here because no missing values are allowed here.
        """
        try:
            return [layer[v] for v in self.__i2v]
        except KeyError:
            msg = "Layers must be specified for all nodes."
            raise ValueError(msg) from None

    def decode(self, i: int) -> _V:
        """Decode the index.

        Returns
        -------
        Value corresponding to the index.

        Raises
        ------
        ValueError
            If `i` is out of range.
        """
        try:
            v = self.__i2v[i]
        except IndexError:
            msg = f"{i} not found."
            raise ValueError(msg) from None
        return v

    def decode_set(self, iset: AbstractSet[int]) -> set[_V]:
        """Decode set."""
        return {self.decode(i) for i in iset}

    def decode_flow(self, f_: Mapping[int, int]) -> dict[_V, _V]:
        """Decode MBQC flow.

        Returns
        -------
        `f_` with both keys and values transformed.
        """
        return {self.decode(i): self.decode(j) for i, j in f_.items()}

    def decode_gflow(self, f_: Mapping[int, AbstractSet[int]]) -> dict[_V, set[_V]]:
        """Decode MBQC gflow.

        Returns
        -------
        `f_` with both keys and values transformed.
        """
        return {self.decode(i): self.decode_set(si) for i, si in f_.items()}

    def decode_layer(self, layer_: Iterable[int]) -> dict[_V, int]:
        """Decode MBQC layer.

        Returns
        -------
        `layer_` transformed.

        Notes
        -----
        `list` (generalized as `Iterable`) is used instead of `dict` here because no missing values are allowed here.
        """
        return {self.decode(i): li for i, li in enumerate(layer_)}

    def decode_err(self, err: ValueError) -> ValueError:
        """Decode the error message stored in the first ctor argument of ValueError."""
        raw = err.args[0]
        # Keep in sync with Rust-side error messages
        if isinstance(raw, FlowValidationMessage.ExcessiveNonZeroLayer):
            node = self.decode(raw.node)
            msg = f"Layer-{raw.layer} node {node} inside output nodes."
        elif isinstance(raw, FlowValidationMessage.ExcessiveZeroLayer):
            node = self.decode(raw.node)
            msg = f"Zero-layer node {node} outside output nodes."
        elif isinstance(raw, FlowValidationMessage.InvalidFlowCodomain):
            node = self.decode(raw.node)
            msg = f"f({node}) has invalid codomain."
        elif isinstance(raw, FlowValidationMessage.InvalidFlowDomain):
            node = self.decode(raw.node)
            msg = f"f({node}) has invalid domain."
        elif isinstance(raw, FlowValidationMessage.InvalidMeasurementSpec):
            node = self.decode(raw.node)
            msg = f"Node {node} has invalid measurement specification."
        elif isinstance(raw, FlowValidationMessage.InconsistentFlowOrder):
            node1 = self.decode(raw.nodes[0])
            node2 = self.decode(raw.nodes[1])
            msg = f"Flow-order inconsistency on nodes ({node1}, {node2})."
        elif isinstance(raw, FlowValidationMessage.InconsistentFlowPlane):
            node = self.decode(raw.node)
            msg = f"Broken {raw.plane} measurement on node {node}."
        elif isinstance(raw, FlowValidationMessage.InconsistentFlowPPlane):
            node = self.decode(raw.node)
            msg = f"Broken {raw.pplane} measurement on node {node}."
        else:
            raise TypeError  # pragma: no cover
        return ValueError(msg)

    def ecatch(self, f: Callable[_S, _T], *args: _S.args, **kwargs: _S.kwargs) -> _T:
        """Wrap binding call to decode raw error messages."""
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            raise self.decode_err(e) from None
