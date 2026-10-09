# Adaptive Context Learning

Version 0.7 adds an optional persistent policy that can improve context selection from repeated use.

## Principle

The deterministic reducer remains the safety boundary. Learning only adjusts the priority of optional context before reduction.

Required context is never demoted by the learner.

~~~text
RAW vs Optimized outcome
        |
        +-- RAW fail -----------------> no learning
        |
        +-- both pass ----------------> slowly lower keep-bias for safely omitted features
        |
        +-- RAW pass / OPT fail ------> strongly increase keep-bias for omitted features
        v
persistent policy.json
~~~

## What is learned

The policy does not store context text. It stores bounded statistics for metadata-derived features:

- source
- state_slot
- normalized key family

Each feature tracks:

- priority bias
- successful omissions
- compression regressions

Bias is bounded so repeated feedback cannot grow without limit.

## Why regression is weighted more strongly

A successful omission is weak evidence that an item class may be safe to remove. A RAW-success / Optimized-failure pair is much stronger evidence that compression removed something important.

The default update is asymmetric:

- successful omission: -1 keep-bias
- compression regression: +8 keep-bias

This makes the policy conservative after failures.

## Training run

Train on the first 25 turns of a 50-turn long workload:

~~~powershell
context-pleum-bench `
  --model gpt-6-luna `
  --long50 `
  --case-start 1 `
  --case-end 25 `
  --token-budget 256 `
  --selection-strategy priority `
  --learn-policy reports/context-policy.json `
  --output reports/learn-train-1-25.json
~~~

The policy file is saved after every RAW/Optimized pair so progress survives interruption.

## Held-out baseline

Run turns 26-50 without a policy:

~~~powershell
context-pleum-bench `
  --model gpt-6-luna `
  --long50 `
  --case-start 26 `
  --case-end 50 `
  --token-budget 256 `
  --selection-strategy priority `
  --output reports/eval-baseline-26-50.json
~~~

## Held-out learned evaluation

Run exactly the same turns with the learned policy frozen:

~~~powershell
context-pleum-bench `
  --model gpt-6-luna `
  --long50 `
  --case-start 26 `
  --case-end 50 `
  --token-budget 256 `
  --selection-strategy priority `
  --policy reports/context-policy.json `
  --output reports/eval-learned-26-50.json
~~~

Compare the two held-out reports. The learned policy is useful only if it improves the token/quality Pareto frontier: fewer tokens at the same quality, or better quality at similar token usage.

## Safety properties

- no raw context text is written to policy.json
- required items are never reprioritized
- RAW failures produce no learning signal
- bias is bounded
- frozen evaluation is supported
- deterministic reducer remains usable with learning completely disabled

## Future learned compressor

Once enough labeled trajectories exist, the next layer can train a small token-classification compressor from successful RAW/Optimized examples and failure recoveries. This is closer to LLMLingua-2 style distillation and should remain separate from the safe deterministic core.
