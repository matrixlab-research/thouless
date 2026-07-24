use std::f64::consts::PI;

use thouless::geometry::ReciprocalPath;
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
