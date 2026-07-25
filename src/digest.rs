//! Deterministic random-access variates derived from immutable byte strings.

use md5::{Digest, Md5};

const FLOAT_BITS: u32 = 53;
const FLOAT_MASK: u64 = (1_u64 << FLOAT_BITS) - 1;
const FLOAT_RECIPROCAL: f64 = 1.0 / (1_u64 << FLOAT_BITS) as f64;

/// Returns two deterministic independent values in the half-open interval
/// `[0, 1)`.
#[must_use]
pub fn uniform_pair(input: &[u8], salt: &[u8]) -> (f64, f64) {
    let mut digest = Md5::new();
    digest.update(input);
    digest.update(salt);
    let bytes = digest.finalize();
    let first = u64::from_ne_bytes(bytes[0..8].try_into().expect("eight-byte digest half"));
    let second = u64::from_ne_bytes(bytes[8..16].try_into().expect("eight-byte digest half"));
    (
        (first & FLOAT_MASK) as f64 * FLOAT_RECIPROCAL,
        (second & FLOAT_MASK) as f64 * FLOAT_RECIPROCAL,
    )
}

/// Returns one deterministic value in the half-open interval `[0, 1)`.
#[must_use]
pub fn uniform(input: &[u8], salt: &[u8]) -> f64 {
    uniform_pair(input, salt).0
}

/// Returns one deterministic standard-normal variate using the Box-Muller
/// transform.
#[must_use]
pub fn gaussian(input: &[u8], salt: &[u8]) -> f64 {
    let (angle, radius) = uniform_pair(input, salt);
    (angle * std::f64::consts::TAU).cos() * (-2.0 * (-radius).ln_1p()).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variates_are_reproducible_bounded_and_salt_sensitive() {
        let first = uniform_pair(b"orbital", b"sample");
        assert_eq!(first, uniform_pair(b"orbital", b"sample"));
        assert!(first.0 >= 0.0 && first.0 < 1.0);
        assert!(first.1 >= 0.0 && first.1 < 1.0);
        assert_ne!(first, uniform_pair(b"orbital", b"different"));
        assert!(gaussian(b"orbital", b"sample").is_finite());
    }
}
