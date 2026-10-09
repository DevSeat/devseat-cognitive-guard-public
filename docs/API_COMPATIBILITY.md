# Public API compatibility

## Version identity

The public distribution version is `1.0.0+public.1`. It is curated from the
verified generic Core `v1.0.0` source boundary and uses a distinct identifier
because one internal-only adapter API is intentionally omitted.

## Deliberately excluded API

The public package does not include:

- `devseat_cognitive_guard.integrations.devseat_operator.OperatorSnapshotAdapter`
- The `OperatorSnapshotAdapter` export from
  `devseat_cognitive_guard.integrations`

The adapter and its two adapter-specific tests were excluded from
`tests/test_product_adapters.py`:

- `test_operator_snapshot_required_sections_survive_tight_budget`
- `test_operator_snapshot_reconstruction_preserves_original_objects`

There is no public replacement for this internal integration. Applications
using the generic `ContextAdapter`, `PortableContextOptimizer`,
`MessageHistoryAdapter`, or BIO/Laya integration retain those APIs. Generic
runtime, policy, failure-replay, CLI, and benchmark APIs remain available as
shown by the source and the test suite.

## Non-API fixture cleanup

The product-isolation test uses `example-bio` and `example-agent` namespaces
instead of internal product names. This changes only test data and does not
change the `PolicyNamespace` or `FilePolicyStore` API.

## Source boundary

This public distribution contains the curated v1.0 source only. It has a new
Git history and does not contain commits, branches, or tags from the private
development repository. Post-v1.0 productization and commercial features are
outside this release.
