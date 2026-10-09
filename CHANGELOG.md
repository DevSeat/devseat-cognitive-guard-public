# Changelog

## 1.0.0+public.1 - 2026-10-09

Curated public distribution based on the verified generic Core `v1.0.0` source
boundary. The `+public.1` identifier distinguishes this distribution from the
original private version because its API intentionally excludes an internal
adapter.

- Froze the model-agnostic context optimizer and runtime contracts.
- Included deterministic context reduction, portable Python and JSONL APIs,
  token-counter injection, adaptive policy support, and failure replay as they
  existed at the tagged v1.0.0 boundary.
- Included the public generic API, 89-test curated suite, and local synthetic
  benchmark workflow.
- Excluded the internal Operator snapshot adapter and its two adapter-specific
  tests. See `docs/API_COMPATIBILITY.md`.
- Added Apache License 2.0 metadata and attribution.

The exact build, installation, test, and benchmark evidence for this candidate
is recorded in `docs/BENCHMARKS.md`.
