//! Common functionalities.

use std::collections::BTreeSet;

use pyo3::{exceptions::PyValueError, prelude::*};
use thiserror::Error;

use crate::{gflow::Plane, pflow::PPlane};

/// Node index.
pub type Node = usize;
/// Layer index.
pub type Layer = usize;
/// Set of nodes.
pub type Nodes = hashbrown::HashSet<Node>;
/// Simple graph encoded as list of neighbors.
pub type Graph = Vec<Nodes>;
/// Layer representation of the flow partial order.
pub type Layers = Vec<Layer>;

/// Ordered set of nodes.
///
/// # Note
///
/// Used only when iteration order matters.
pub(crate) type OrderedNodes = BTreeSet<Node>;

/// Error type for flow validation.
///
/// Python name does not contain `Error` as it is not a subclass of `Exception`.
#[pyclass(name = "FlowValidationMessage")]
#[derive(Debug, Error, PartialEq, Eq, Clone, Hash)]
pub enum FlowValidationError {
    // Keep in sync with Python-side error messages
    #[error("layer-{layer} node {node} inside output nodes")]
    ExcessiveNonZeroLayer { node: Node, layer: Layer },
    #[error("zero-layer node {node} outside output nodes")]
    ExcessiveZeroLayer { node: Node },
    #[error("f({node}) has invalid codomain")]
    InvalidFlowCodomain { node: Node },
    #[error("f({node}) has invalid domain")]
    InvalidFlowDomain { node: Node },
    #[error("node {node} has invalid measurement specification")]
    InvalidMeasurementSpec { node: Node },
    #[error("flow-order inconsistency on nodes ({}, {})",.nodes.0, .nodes.1)]
    InconsistentFlowOrder { nodes: (Node, Node) },
    #[error("broken {plane:?} measurement on node {node}")]
    InconsistentFlowPlane { node: Node, plane: Plane },
    #[error("broken {pplane:?} measurement on node {node}")]
    InconsistentFlowPPlane { node: Node, pplane: PPlane },
}

impl From<FlowValidationError> for PyErr {
    #[inline]
    fn from(e: FlowValidationError) -> Self {
        PyValueError::new_err(e)
    }
}

// TODO: Remove once stabilized
pub const FATAL_MSG: &str = "\
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!            POST VERIFICATION FAILED            !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Please report to the developers via GitHub:
https://github.com/TeamGraphix/swiflow/issues/new";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_err_from() {
        let _ = PyErr::from(FlowValidationError::ExcessiveNonZeroLayer { node: 1, layer: 2 });
    }
}
