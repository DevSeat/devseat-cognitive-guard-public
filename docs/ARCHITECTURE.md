# Architecture

DevSeat Cognitive Guard public release `1.0.0+public.1` reduces assembled
context before an application sends it to a model. It does not call a model
from the core optimizer or rewrite the user's prompt. The distribution is
curated from the generic Core v1.0.0 source boundary; its public API exclusions
are recorded in `API_COMPATIBILITY.md`.

## Data flow

1. A caller provides `ContextItem` values and an optional character or token
   budget.
2. `ContextOptimizer` delegates deterministic selection and reduction to
   `ContextReducer`.
3. Required items, fingerprints, omissions, and reduction statistics are
   returned in `OptimizedContext` / `ReducedContext`.
4. `PortableContextOptimizer` and `ContextAdapter` let an application convert
   structured payloads to and from the generic context representation.
5. The JSONL bridge exposes the portable API to another process without a
   Python import boundary.

The default token counter is a dependency-free estimate. Callers can inject an
exact host tokenizer through the `TokenCounter` interface. Optional policy
storage keeps learned selection metadata separate from context text. Failure
replay can attribute some omission regressions using bounded probes.

## Boundaries

- The library has no runtime package dependencies at `1.0.0+public.1`.
- It does not hold provider credentials or make model calls during ordinary
  optimization.
- The PleumRouter benchmark commands are separate tools and make live network
  requests only when explicitly run.
- Context reduction is lossy when optional items are omitted. Callers decide
  which items are required and should evaluate quality for their own workload.
- Optional adapters live under `integrations/`; callers can use the portable
  API directly without importing them.
