# Contributing

Contributions should preserve the model-agnostic core and deterministic default
behavior. Keep provider-specific integrations optional and avoid adding
credentials, private customer data, or copied third-party material.

This project is licensed under Apache-2.0. Unless a contribution explicitly
states otherwise, contributions submitted for inclusion are offered under the
terms in Section 5 of `LICENSE`.

## Development checks

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
python benchmarks/synthetic_context_benchmark.py --cases 100
```

Include the Python version and exact commands with a change. Live-provider
benchmarks are optional, may incur cost, and must use credentials supplied
through environment variables.
