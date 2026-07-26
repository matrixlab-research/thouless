# Third-party native backends

Thouless uses a platform-specific LAPACK backend:

- Apple Accelerate on macOS;
- Intel oneMKL, linked statically in sequential LP64 mode, on other supported
  platforms.

The oneMKL backend is distributed under the
[Intel Simplified Software License](https://www.intel.com/content/www/us/en/developer/articles/license/end-user-license-agreement.html).
The Rust source remains licensed under `LICENSE-MIT`; binary redistributors
must also comply with the license of the native backend included in their
artifact.
