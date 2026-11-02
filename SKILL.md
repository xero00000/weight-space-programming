---
name: weight-space-programming
description: Inspect, explain, repair, or construct neural-network weights directly through reasoning and tensor edits without training. Use for weight-space interpretation, targeted checkpoint surgery, reasoning-loop diagnosis, and hand-written neural policies or algorithmic transformer circuits. Do not use for ordinary fine-tuning, inference optimization, or prompt tuning.
---

# Weight-Space Programming

Translate a behavioral specification into an explanation or a concrete tensor intervention. Use the methods in [Takehi et al. (2026)](https://rikiyat.github.io/blog/weight-space-programming/), while treating interpretations as hypotheses and improvements as measurements. A skill supplies a method; it does not confer the capabilities reported for the paper's editor.

## Establish the experiment

Identify the checkpoint revision and format, architecture and tensor orientation, input/output semantics, observed failure or desired behavior, and available compute. Check model code and configuration before reasoning about tensor names. Preserve the original checkpoint and create candidates separately.

Choose the evidence regime from the user's request:

- **Weights only:** inspect tensors and algebraic compositions; do not execute the network, individual layers on inputs, or synthetic probes. Separate weight-space evidence from behavioral speculation. If only a subset is visible, describe the subset.
- **Bounded probes:** use a declared budget for generations, activations, or environment episodes. Count examples, batch sizes, tokens and calls, since a paper's “forward pass” may contain hundreds of sequences. Keep the same decoding and environment settings for paired comparisons.
- **Construction:** derive a small algorithm from the supplied rules and encode it in the specified architecture. Declare whether changing dimensions, normalization, attention, or module types is permitted. Respect zero-feedback restrictions when requested.

Do not quietly relax an evidence restriction. If the task leaves budgets open, choose a small explicit initial budget compatible with the available hardware; do not start a large benchmark or download a large model merely to use the skill.

## Route to the relevant method

Read [references/methods.md](references/methods.md) for the chosen mode:

- **Read:** map input geometry, output structure, or weight compositions to a testable interpretation. Correlation and matrix structure alone do not establish behavior.
- **Repair:** distinguish a bottleneck such as repetition from missing capability. Localize the writer, propose the smallest interpretable intervention, and compare it against the unchanged baseline and suitable controls.
- **Construct:** assign units to features and operations, propagate their values across layers, and reason through nonlinearities and output selection. For algorithmic transformers, read [references/transformer-construction.md](references/transformer-construction.md) for routing/counting circuits, ReLU lookups, architecture constraints and executable examples.

For the article's Qwen3.5-0.8B loop correction, read [references/qwen-reproduction.md](references/qwen-reproduction.md). It points to revision-pinned, hash-checked released edits. Layer numbers, columns, directions and scales from that experiment are not a general recipe for other checkpoints.

## Make edits reviewable

For each candidate record the mechanism hypothesis, exact tensor names/shapes, selected indices, equation, strength, parent checkpoint hashes, and why the change should affect the failure. Direct edits may use tensor algebra under `torch.no_grad()`; exclude backpropagation, optimizers, distillation, and fitting predictor weights to labeled examples, including closed-form regression or label-derived lookup tables. Activation-based localization and bounded feedback on reasoned interventions remain allowed when probes are allowed; distinguish these from training a predictor or black-box parameter search. Task-derived truth tables are construction, while memorizing evaluation answers is not.

Use `scripts/weights.py` for deterministic bookkeeping on **one safetensors file or shard at a time**. Read [references/tools.md](references/tools.md) for commands, memory requirements and the precise projection convention. It can:

- inventory tensor shapes and numeric health;
- compare checkpoints with matching tensor names, shapes and dtypes;
- preview or write direction removal from explicitly selected columns, preserving the source and tensor metadata and producing a hash manifest.

It does not discover a behavioral direction, run inference, convert GGUF/quantized checkpoints, or assemble a complete model directory. Derive candidate directions and selected neurons from the actual evidence. Inspect the stored result after dtype rounding. Count changed scalar values, not just target tensors; specify whether percentages use the edited shard or the full model as denominator. Preserve configuration, tokenizer, chat template and weight index when assembling a separately loadable candidate model. Never edit hard-linked source shards in place.

## Decide from evidence

Use practice/development examples for diagnosis and selection. Freeze the candidate before running the held-out evaluation; do not tune against the reported test set. Compare matched prompts/seeds and identical sampling, token limits, grading and environment settings. Record regressions and uncertainty, especially with small or stochastic samples.

For finite algorithmic tasks, use exhaustive checks when feasible; otherwise include fresh evaluation seeds and boundary cases after freezing the construction. Distinguish sampled accuracy from exhaustive domain coverage and from a mathematical proof. When comparing size, count frozen parameters, registered buffers, lookup tables and constants stored in code as well as trainable parameters, and disclose architecture changes. Moving constants to buffers does not simplify the underlying mechanism.

For reasoning loops, measure answer accuracy, repeated spans, token-limit hits, completion length, and whether thinking ends and a usable final answer appears. A shorter answer alone is not a successful repair. Check already-successful examples and an unrelated behavior such as instruction following. For policies, compare episode success and reward under the same action sampling and seed distribution.

Retain an edit only if it meets the declared success criterion without unacceptable regressions. Revert rejected candidates by selecting the original checkpoint. Stop at the agreed budget or when evidence no longer supports a localized fix; “no established improvement” is a valid result. Do not claim new reasoning ability when the edit only permits existing reasoning to finish.

Return the candidate location (or the unchanged baseline), reproducible patch/manifest, mechanism and its evidence, evaluation settings and counts, baseline/candidate results, limitations and rollback path. Separate source-reported numbers from anything measured locally. See [references/sources.md](references/sources.md) for attribution and reported limitations.
