# Development Workflow

## Requirements

- Python 3.10 or newer.
- pip and a working C compiler are not required for the core package.
- The runtime package has no third-party dependencies. Pytest is used for
  development checks.

## Install and check

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest -q
context-reduce --help
python benchmarks/synthetic_context_benchmark.py --cases 100
```

The installed `context-reduce` command accepts JSON input and can write the
reduced result to a file. The `context-bridge` command reads and writes JSONL
messages on standard input and output.

## Continuous integration

`.github/workflows/ci.yml` runs the test suite and synthetic benchmark on the
supported Python matrix. A local run does not prove that GitHub Actions has
executed successfully; check the workflow run on the target repository before
release.
