# Agent Instructions

This file applies to the entire repository.

Before planning, designing an API, writing implementation code, changing tests,
or reporting project status, read and follow:

- [`instructions/scientific-software-reimplementation.md`](instructions/scientific-software-reimplementation.md)

That document is the authoritative repository instruction for the Thouless
reimplementation. A nested `AGENTS.md` may add stricter component-specific
rules, but it must not weaken or contradict the root instruction.

User and system instructions take precedence when they explicitly conflict with
this file. Otherwise, all requirements in the linked instruction are mandatory.

At minimum, every contribution must:

1. Implement or validate a general scientific capability rather than fit known
   tests.
2. Keep the native and compatibility coverage matrices current.
3. Link every missing capability, compatibility gap, failing test, error, or
   intentional skip to a reproducible GitHub issue.
4. Preserve upstream comparison rules and numerical tolerances unless the user
   explicitly approves a change.
5. Keep the overall project status `Incomplete` while any required capability,
   interface, public validation, or held-out validation remains open.
6. Run the relevant checks documented in `README.md` before declaring a change
   ready.
7. Extend public interfaces from first principles using the smallest
   user-friendly abstraction that covers the complete scientific requirement
   and generalizes across real workflows. Do not add one public API per
   benchmark, test, or example, and do not expose internal AD machinery or
   solver bookkeeping on the ordinary user path. A concept belongs in the
   public interface only when it has a clear scientific meaning or is reusable
   across multiple real workflows; keep advanced controls available without
   making them prerequisites for common tasks.
