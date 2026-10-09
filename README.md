# DevSeat Cognitive Guard

DevSeat Cognitive Guard is a deterministic, model-agnostic Python library for
reducing assembled context before an application sends it to a language model.
The public curated release is `1.0.0+public.1`, based on the generic Core
v1.0.0 source boundary. The core optimizer does not call a model or rewrite
prompts. See [public API compatibility](docs/API_COMPATIBILITY.md) for the
intentional exclusion of the internal Operator adapter.

## Install

Python 3.10 or newer is required. The core has no runtime dependencies.

```powershell
python -m pip install .
```

Install development tools with:

```powershell
python -m pip install ".[dev]"
```

## Python API

```python
from devseat_cognitive_guard import ContextItem, ContextOptimizer, ContextRequest

result = ContextOptimizer().optimize(
    ContextRequest(
        context=(
            ContextItem(
                key="current-contract",
                content="The service must retain the public API.",
                required=True,
                priority=100,
            ),
            ContextItem(
                key="old-log",
                content="A historical build completed successfully.",
                priority=5,
            ),
        ),
        context_budget_tokens=128,
    )
)

print(result.reduced_context.items)
print(result.saved_context_tokens)  # estimated unless an exact counter is injected
```

The built-in token counter is a dependency-free estimate. Applications can
inject a host tokenizer through the `TokenCounter` interface. Required items
are protected from budget omission; callers are responsible for marking the
context that must be retained.

## CLI

Install the package, then run:

```powershell
context-reduce input.json --output reduced.json
context-reduce --help
```

The JSONL `context-bridge` command is available for process-isolated
integrations. See [docs/CONTEXT_OPTIMIZER.md](docs/CONTEXT_OPTIMIZER.md) and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Tests and benchmarks

```powershell
python -m pytest -q
python benchmarks/synthetic_context_benchmark.py --cases 100 --noise-blocks 20
```

The synthetic benchmark is local and makes no model calls. Its token reduction
uses the package's approximate counter and is not evidence of answer quality on
real workloads. Live-provider benchmark commands are documented separately in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md); they can make paid network requests
to the public PleumRouter API and require a user-supplied key in the
`PLEUM_API_KEY` environment variable.

## Project documents

- [Architecture](docs/ARCHITECTURE.md)
- [Development workflow](docs/DEVELOPMENT_WORKFLOW.md)
- [Benchmark methodology](docs/BENCHMARKS.md)
- [Public API compatibility](docs/API_COMPATIBILITY.md)
- [Security policy](SECURITY.md)
- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)
- [License](LICENSE) (Apache-2.0; copyright notice in [NOTICE](NOTICE))
