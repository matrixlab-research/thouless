# Public language-conformance cases

The parity job generates the same scientific cases independently through the
stable Rust, Python, and Julia interfaces:

- an SSH chain with an analytic gap and quantized reduced polarization;
- a Qi-Wu-Zhang Chern insulator;
- a finite vacancy geometry and projected local observable;
- a matched lead-device-lead system with unit transmission;
- a gauge-transformed Wilson line;
- an invalid momentum shape that must remain a recoverable language error.

The runners emit physical invariants, not stored solver snapshots.
`tools/compare_conformance.py` checks analytic values and cross-language
agreement at operation-specific tolerances. Evaluator-owned held-out inputs
remain outside this repository.
