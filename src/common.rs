//! Common functionalities.

use std::collections::{BTreeSet, HashSet};

use pyo3::{exceptions::PyValueError, prelude::*};
use thiserror::Error;

use crate::{
    common::FlowValidationError::{
        ExcessiveNonZeroLayer, ExcessiveZeroLayer, InvalidFlowCodomain, InvalidFlowDomain,
    },
    gflow::Plane,
    pflow::PPlane,
};

/// Node index.
pub type Node = usize;
/// Layer index.
pub type Layer = usize;
/// Set of nodes.
pub type Nodes = HashSet<Node>;
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

/// Checks if the layer-zero nodes are correctly chosen.
///
/// This check can be skipped unless maximally-delayed flow is required.
///
/// # Arguments
///
/// - `layers`: The layer.
/// - `oset`: The set of output nodes.
/// - `iff`: If `true`, `layers[u] == 0` "iff" `u` is in `oset`. Otherwise "if".
pub fn check_initial(layers: &[Layer], oset: &Nodes, iff: bool) -> Result<(), FlowValidationError> {
    for (u, &lu) in layers.iter().enumerate() {
        match (oset.contains(&u), lu == 0) {
            (true, false) => {
                Err(ExcessiveNonZeroLayer { node: u, layer: lu })?;
            }
            (false, true) if iff => {
                Err(ExcessiveZeroLayer { node: u })?;
            }
            _ => {}
        }
    }
    Ok(())
}

/// Checks if the domain of `f` is in `vset - oset` and the codomain is in `vset - iset`.
///
/// # Arguments
///
/// - `f_flatiter`: Flow, gflow, or pflow as `impl Iterator<Item = (&Node, &Node)>`.
/// - `vset`: All nodes.
/// - `iset`: Input nodes.
/// - `oset`: Output nodes.
pub fn check_domain<'a, 'b>(
    f_flatiter: impl Iterator<Item = (&'a Node, &'b Node)>,
    vset: &Nodes,
    iset: &Nodes,
    oset: &Nodes,
) -> Result<(), FlowValidationError> {
    let icset = vset - iset;
    let ocset = vset - oset;
    let mut dom = Nodes::new();
    for (&i, &fi) in f_flatiter {
        dom.insert(i);
        if !icset.contains(&fi) {
            Err(InvalidFlowCodomain { node: i })?;
        }
    }
    if let Some(&i) = dom.symmetric_difference(&ocset).next() {
        Err(InvalidFlowDomain { node: i })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use core::iter;
    use std::collections::HashMap;

    use super::*;
    use crate::common::Nodes;

    #[test]
    fn test_err_from() {
        let _ = PyErr::from(FlowValidationError::ExcessiveNonZeroLayer { node: 1, layer: 2 });
    }

    #[test]
    fn test_check_initial() {
        let layers = vec![0, 0, 0, 1, 1, 1];
        let oset = Nodes::from([0, 1]);
        check_initial(&layers, &oset, false).unwrap();
    }

    #[test]
    fn test_check_initial_ng() {
        let layers = vec![0, 0, 0, 1, 1, 1];
        let oset = Nodes::from([0, 1, 2, 3]);
        assert!(check_initial(&layers, &oset, false).is_err());
    }

    #[test]
    fn test_check_initial_iff() {
        let layers = vec![0, 0, 0, 1, 1, 1];
        let oset = Nodes::from([0, 1, 2]);
        check_initial(&layers, &oset, true).unwrap();
    }

    #[test]
    fn test_check_initial_iff_ng() {
        let layers = vec![0, 0, 0, 1, 1, 1];
        let oset = Nodes::from([0, 1]);
        assert!(check_initial(&layers, &oset, true).is_err());
    }

    #[test]
    fn test_check_domain_flow() {
        let f = HashMap::<Node, Node>::from([(0, 1), (1, 2)]);
        let vset = Nodes::from([0, 1, 2]);
        let iset = Nodes::from([0]);
        let oset = Nodes::from([2]);
        check_domain(f.iter(), &vset, &iset, &oset).unwrap();
    }

    #[test]
    fn test_check_domain_gflow() {
        let f = HashMap::<Node, Nodes>::from([(0, Nodes::from([1, 2])), (1, Nodes::from([2]))]);
        let vset = Nodes::from([0, 1, 2]);
        let iset = Nodes::from([0]);
        let oset = Nodes::from([2]);
        let f_flatiter = f
            .iter()
            .flat_map(|(i, fi)| Iterator::zip(iter::repeat(i), fi.iter()));
        check_domain(f_flatiter, &vset, &iset, &oset).unwrap();
    }

    #[test]
    fn test_check_domain_ng_iset() {
        let f = HashMap::<Node, Nodes>::from([(0, Nodes::from([0, 1])), (2, Nodes::from([2]))]);
        let vset = Nodes::from([0, 1, 2]);
        let iset = Nodes::from([0]);
        let oset = Nodes::from([2]);
        let f_flatiter = f
            .iter()
            .flat_map(|(i, fi)| Iterator::zip(iter::repeat(i), fi.iter()));
        assert!(check_domain(f_flatiter, &vset, &iset, &oset).is_err());
    }

    #[test]
    fn test_check_domain_ng_oset() {
        let f = HashMap::<Node, Nodes>::from([(0, Nodes::from([1])), (1, Nodes::from([0]))]);
        let vset = Nodes::from([0, 1, 2]);
        let iset = Nodes::from([0]);
        let oset = Nodes::from([2]);
        let f_flatiter = f
            .iter()
            .flat_map(|(i, fi)| Iterator::zip(iter::repeat(i), fi.iter()));
        assert!(check_domain(f_flatiter, &vset, &iset, &oset).is_err());
    }
}
