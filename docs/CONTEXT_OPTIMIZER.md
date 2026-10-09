# Context Optimizer

## Purpose

Reduce context before a model call without rewriting the prompt and without calling another model.

## Scope

The core receives structured context items and returns a smaller structured context set.

```text
ContextItem[]
  -> exact dedup
  -> required preservation
  -> optional priority pruning
  -> fingerprint / delta reuse
  -> ReducedContext
```

No model-specific request format is produced.

## Context item

Each item can carry:

- `key`
- `content`
- `priority`
- `required`
- `stable`
- `source`
- `revision`

## Deduplication

Content is normalized for whitespace and hashed. Exact duplicates are collapsed.

When duplicate items have different metadata, the survivor is selected in this order:

1. required;
2. higher priority;
3. stable;
4. earlier input position.

This prevents a weaker duplicate from replacing mandatory context.

## Budget pruning

Required items are retained even if they exceed the requested budget.

Optional items are sorted by priority and retained while they fit inside the remaining character budget. Items that do not fit are listed in `omitted_keys`.

## Fingerprints

Each context item receives a deterministic fingerprint derived from:

- normalized content;
- source;
- revision;
- stable/dynamic classification.

The fingerprint map is returned to the caller.

## Delta context

Delta mode is opt-in. An item is omitted as unchanged only when:

- it is marked `stable`;
- the caller supplies a previous fingerprint for the same key;
- that fingerprint exactly matches the current item.

A content, source, revision, or stable/dynamic change invalidates reuse.

## Output

`OptimizedContext` contains:

- reduced context items;
- duplicate keys;
- omitted keys;
- unchanged stable keys;
- fingerprint map;
- original/reduced character counts;
- reduction ratio;
- diagnostic notes.

## Provider neutrality

The optimizer has no GPT, Claude, OpenAI, Anthropic, prompt, cache, reasoning-effort, or tool-call logic. Callers decide how to pass the retained context to their model.


## Token-first budgeting

Version 0.4 adds model-independent token accounting.

If `context_budget_tokens` is provided, token cost becomes the primary optional-context budget. The older character budget remains available as a fallback.

The optimizer reports:

- `original_context_tokens`
- `reduced_context_tokens`
- `saved_context_tokens`
- `token_reduction_ratio`

The default counter is intentionally approximate and conservative. For exact provider/model accounting, inject a tokenizer through the `TokenCounter` interface.

This keeps the reduction algorithm independent from GPT, Claude, or any single tokenizer implementation.
