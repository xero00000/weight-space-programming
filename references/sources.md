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

The skill's helper functions are engineering utilities for inspection, projection and comparison. Passing their tests validates those operations, not the article's empirical claims.
