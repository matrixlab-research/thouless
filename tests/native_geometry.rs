use std::f64::consts::PI;

use thouless::geometry::{ReciprocalPath, UniformReciprocalMesh};
use thouless::model::Lattice;
use thouless::GeometryError;

#[test]
fn reciprocal_path_uses_the_cartesian_reciprocal_metric() {
    let lattice = Lattice::new(vec![vec![2.0, 0.0], vec![0.0, 1.0]], vec![0, 1]).unwrap();
    let nodes = vec![vec![0.0, 0.0], vec![0.5, 0.0], vec![0.5, 0.5]];
    let path = ReciprocalPath::through(&lattice, &nodes, 7).unwrap();

    assert_eq!(path.reduced_points().len(), 7);
    assert_eq!(path.reduced_points().first().unwrap(), &nodes[0]);
    assert_eq!(path.reduced_points().last().unwrap(), &nodes[2]);
    assert!((path.node_distances()[1] - 0.5 * PI).abs() < 1.0e-12);
    assert!((path.node_distances()[2] - 1.5 * PI).abs() < 1.0e-12);
    assert!((path.distances().last().unwrap() - 1.5 * PI).abs() < 1.0e-12);
}

#[test]
fn reciprocal_path_rejects_under_specified_sampling() {
    let lattice = Lattice::new(vec![vec![1.0]], vec![0]).unwrap();
    let error =
        ReciprocalPath::through(&lattice, &[vec![0.0], vec![0.5], vec![1.0]], 2).unwrap_err();
    assert_eq!(
        error,
        GeometryError::InsufficientPathSamples {
            minimum: 3,
            actual: 2
        }
    );
}

#[test]
fn short_path_segments_retain_every_requested_node() {
    let lattice = Lattice::new(vec![vec![1.0]], vec![0]).unwrap();
    let nodes = vec![vec![0.0], vec![1.0e-12], vec![1.0]];
    let path = ReciprocalPath::through(&lattice, &nodes, 3).unwrap();

    assert_eq!(path.reduced_points(), nodes);
    assert_eq!(path.node_distances().len(), 3);
}

#[test]
fn uniform_mesh_has_stable_order_coordinates_and_quadrature_measure() {
    let lattice = Lattice::new(vec![vec![2.0, 0.0], vec![0.5, 1.0]], vec![0, 1]).unwrap();
    let mesh = UniformReciprocalMesh::new(&lattice, &[2, 3], &[0.5, 0.0]).unwrap();

    assert_eq!(mesh.shape(), &[2, 3]);
    assert_eq!(mesh.fractional_offsets(), &[0.5, 0.0]);
    assert_eq!(mesh.reduced_points().len(), 6);
    assert_eq!(mesh.reduced_points()[0], vec![0.25, 0.0]);
    assert_eq!(mesh.reduced_points()[1], vec![0.25, 1.0 / 3.0]);
    assert_eq!(mesh.reduced_points()[3], vec![0.75, 0.0]);
    assert!((mesh.cartesian_points()[0][0] - PI / 4.0).abs() < 1.0e-12);
    assert!((mesh.cartesian_points()[0][1] + PI / 8.0).abs() < 1.0e-12);
    assert!((mesh.reciprocal_volume() - 2.0 * PI * PI).abs() < 1.0e-12);
    assert!((mesh.normalized_weight() - 1.0 / 6.0).abs() < 1.0e-12);
    assert!((mesh.cartesian_weight() - PI * PI / 3.0).abs() < 1.0e-12);
}

#[test]
fn lower_dimensional_mesh_uses_the_embedded_reciprocal_measure() {
    let lattice = Lattice::new(vec![vec![3.0, 4.0], vec![-4.0, 3.0]], vec![0]).unwrap();
    let mesh = UniformReciprocalMesh::cell_centered(&lattice, &[2]).unwrap();

    assert_eq!(mesh.reduced_points(), &[vec![0.25], vec![0.75]]);
    assert!((mesh.reciprocal_volume() - 2.0 * PI / 5.0).abs() < 1.0e-12);
    assert!((mesh.cartesian_weight() - PI / 5.0).abs() < 1.0e-12);
}

#[test]
fn uniform_mesh_rejects_ambiguous_or_empty_axes() {
    let lattice = Lattice::new(vec![vec![1.0, 0.0], vec![0.0, 1.0]], vec![0, 1]).unwrap();
    assert_eq!(
        UniformReciprocalMesh::gamma_centered(&lattice, &[4]).unwrap_err(),
        GeometryError::InvalidMeshShape {
            expected: 2,
            actual: 1
        }
    );
    assert_eq!(
        UniformReciprocalMesh::gamma_centered(&lattice, &[4, 0]).unwrap_err(),
        GeometryError::EmptyMeshAxis { axis: 1 }
    );
    assert_eq!(
        UniformReciprocalMesh::new(&lattice, &[4, 4], &[0.0, 1.0]).unwrap_err(),
        GeometryError::InvalidMeshOffset { axis: 1 }
    );
}
