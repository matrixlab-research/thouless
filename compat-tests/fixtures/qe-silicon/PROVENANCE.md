# Quantum ESPRESSO silicon bands fixture provenance

`si_bands.dat` is the unmodified content of the official Quantum ESPRESSO
silicon `bands.x` reference output. It was renamed from `sibands.dat` so that
the PythTB-compatible `read_bands_qe(root, "si")` naming convention can read
it directly.

- source repository: <https://github.com/QEF/q-e>
- release tag: `qe-7.6`
- source commit: `9f93ddec427d2b9a45bb72d828c6d324f62fcabd`
- original path: `PP/examples/example01/reference/sibands.dat`
- Git blob identifier: `36dd92dcf4944eede2728146fea409e07dee7d15`
- SHA-256: `048b9e210c5a6a216e3809385ca62670b01ff8ceee19a9a315613f45c568b22a`
- dimensions declared by the output: 72 k-points and 8 bands

The example input uses an fcc silicon cell with `ibrav=2`,
`celldm(1)=10.2` bohr, and a `tpiba_b` path from L to Gamma to X to K to
Gamma. Its conventional lattice parameter is 5.3976 angstrom.

`LICENSE.txt` contains the repository's GNU General Public License version 2
text. The source file contains Latin-1 copyright bytes; it was normalized to
UTF-8 for this text-only repository patch. The license wording is unchanged.
