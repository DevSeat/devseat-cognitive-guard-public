# Failure Replay and Adaptive Attribution

Version 0.8 adds bounded replay after a compression regression.

## Why

Before v0.8, a RAW-success / Optimized-failure pair increased keep-bias for every omitted feature class. That is safe but noisy: one critical item can cause many unrelated omitted items to receive positive bias.

Failure replay narrows the cause before learning.

~~~text
RAW PASS / OPT FAIL
        |
        v
restore all omitted items
        |
        +-- still FAIL -> unresolved, fall back to broad conservative learning
        |
        v
      PASS
        |
        v
bisect omitted set
        |
        +-- left alone recovers -> recurse left
        +-- right alone recovers -> recurse right
        +-- neither half recovers -> interaction group
        v
small attributed key set
        |
        v
update only those feature buckets
~~~

## Safety / cost controls

- bounded by --max-replay-probes
- only runs after RAW succeeds and Optimized fails
- retries still apply to replay requests
- attributed dataset does not store raw context text
- unresolved failures fall back to the conservative broad update
- multi-item interactions are preserved as a group instead of forcing one-item attribution

## Run

~~~powershell
context-pleum-bench `
  --model gpt-6-luna `
  --learning-challenge `
  --challenge-turns 12 `
  --token-budget 200 `
  --learn-policy reports/replay-policy.json `
  --failure-replay `
  --max-replay-probes 8 `
  --replay-dataset reports/failure-replay.jsonl `
  --output reports/replay-train.json
~~~

The terminal total report adds:

- replay probe count
- number of regressions recovered by replay
- number of attributed keys

## Generalization challenge

The generalization workload changes the critical evidence source halfway through while keeping the key family stable. It checks whether learning transfers beyond one memorized source name.

Train on turns 1-6:

~~~powershell
context-pleum-bench `
  --model gpt-6-luna `
  --generalization-challenge `
  --challenge-turns 12 `
  --source-switch-turn 7 `
  --case-start 1 `
  --case-end 6 `
  --token-budget 200 `
  --learn-policy reports/generalization-policy.json `
  --failure-replay `
  --replay-dataset reports/generalization-replay.jsonl `
  --output reports/generalization-train.json
~~~

Evaluate unseen source turns 7-12 with the policy frozen:

~~~powershell
context-pleum-bench `
  --model gpt-6-luna `
  --generalization-challenge `
  --challenge-turns 12 `
  --source-switch-turn 7 `
  --case-start 7 `
  --case-end 12 `
  --token-budget 200 `
  --policy reports/generalization-policy.json `
  --output reports/generalization-eval.json
~~~

## Dataset role

The replay JSONL is intended as future supervision for a small learned compressor. It stores case IDs, attributed keys/features, recovery status, probe counts, and budget metadata, but not context bodies.
