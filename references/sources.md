# Sources and attribution

This skill is an independent implementation and methodological adaptation. It is not affiliated with the researchers and does not bundle their logs, networks, datasets or patches.

Primary source: Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab, **Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights**, November 2, 2026. [Article](https://rikiyat.github.io/blog/weight-space-programming/).

Released evidence: [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs), reviewed at commit `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f` on November 2, 2026. The repository contains experiment prompts, responses, code, edits and evaluation records. Its license distinguishes CC BY 4.0 logs/data/networks, MIT author-written scripts, GPL MinAtar source quoted in prompts, and separately licensed third-party datasets. Consult the [upstream license](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/LICENSE) before redistributing upstream artifacts. This repository's MIT license covers its original implementation; upstream artifacts retain their own terms.

## Boundaries learned from the experiments

- Small spatial/game networks were amenable to first/last-layer geometry, but this does not establish general behavioral recovery from large weights.
- The diffusion interpretation recognized shapes but missed that they were filled.
- Transformer copying heads could be inferred in the tested character models through weight analysis; architecture and task assumptions matter.
- Sycophancy could not be distinguished from weights alone in the reported attempts; attention-only probes enabled the reported identification.
- The Qwen result primarily removed repetitive thinking that prevented completion. The non-looping Gemma follow-up did not establish a math gain.
- A paper's budget unit can be a batched generation/evaluation request rather than one neural forward computation. Experiment 8 also permitted teacher-forced activation reads.
- Sparse edits can have broad effects. A small touched fraction does not establish that unrelated capabilities are preserved.

## Algorithmic transformer construction

Chandan Singh, **Weight-Space Programming from scratch on simple tasks**, November 2, 2026. [Report](https://csinva.io/weight-space-programming-autoresearch/analyze/toy_examples/report.html). Reviewed with [source snapshot `535cd6b1d881b7ce7124858244b768a63d6d2d5d`](https://github.com/csinva/weight-space-programming-autoresearch/tree/535cd6b1d881b7ce7124858244b768a63d6d2d5d), including the [agent instructions](https://github.com/csinva/weight-space-programming-autoresearch/blob/535cd6b1d881b7ce7124858244b768a63d6d2d5d/evolve/program.md) and [report source](https://github.com/csinva/weight-space-programming-autoresearch/blob/535cd6b1d881b7ce7124858244b768a63d6d2d5d/analyze/toy_examples/report.html).

The report motivates the construction circuits and evaluation/accounting guidance in [transformer-construction.md](transformer-construction.md). It describes routing/counting attention, finite-domain MLP tables, residual-channel allocation and arithmetic mechanisms. It also identifies invalid language-task results from evaluation leakage and prohibited fitting, and misleading parameter reductions from constants moved into buffers. Fixed-seed task scores require independent verification, and architecture changes limit size comparisons. We retain bounded experiments rather than the upstream prompt's indefinite iteration policy.

The examples in this skill are original, small implementations of these general circuit ideas; no upstream program, weights or dataset is bundled. Their exhaustive tests establish only their stated finite-domain behavior and tested dtypes. They do not reproduce the report's full experiment collection or validate its language-task scores.

The checkpoint helpers are engineering utilities for inspection, projection and comparison. Passing their tests validates those operations, not either article's empirical claims.
