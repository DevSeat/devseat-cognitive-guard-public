# Benchmarks

## Local synthetic benchmark

Run the dependency-free synthetic workload after installing the package:

```powershell
python benchmarks/synthetic_context_benchmark.py --cases 100 --noise-blocks 20
```

The benchmark generates artificial context items, runs the local optimizer,
and reports estimated token counts, required-item retention, and elapsed wall
time. It makes no network requests and does not call a model. Required-item
retention is a structural check, not a measure of answer quality. Wall time is
machine-dependent and should not be compared across unlike environments.

## Live-provider benchmarks

The tagged `context-pleum-bench`, `context-baseline-bench`,
`context-hybrid-bench`, `context-agentic-bench`, `context-execution-bench`, and
`context-runtime-bench` commands call PleumRouter and can incur provider cost.
They require an API key in `PLEUM_API_KEY`; never put a real key in a fixture or
report. Their input-token totals come from the provider response, while some
stage-level estimates use the local token counter. Read the CLI help and review
the planned request count before running one.

The public suite contains 89 tests after excluding two tests for the internal
Operator adapter. Run the test command from the repository root to reproduce
the local suite; the checked-in GitHub Actions workflow tests Python 3.10 and
3.13 for each published commit.

## Interpretation

Synthetic workloads test deterministic mechanics and are not evidence of
quality or savings on real user requests. Live-provider results are tied to the
model, provider, workload, date, and request settings used. Report those
conditions with any result and do not generalize a single run to other tasks.
