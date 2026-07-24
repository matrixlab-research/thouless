# Contributing

Every change should add or repair a general scientific capability.

## Required evidence

1. Update the native or compatibility coverage matrix.
2. Add direct Rust tests for native behavior.
3. Add or enable source-interface tests when a compatibility entry changes.
4. Preserve existing upstream comparison rules and numerical tolerances.
5. Link every remaining missing capability, failure, error, or intentional skip
   to an open GitHub issue.

Do not branch on test names, paths, fixtures, input fingerprints, or execution
order. Do not embed expected outputs or weaken comparisons to make a known test
pass.

A green subset is not completion. The project remains **Incomplete** while any
capability, in-scope source interface, designated source test, or held-out
validation item remains open.
