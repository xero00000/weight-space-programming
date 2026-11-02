# Weight-Space Programming

An agent skill for reading, repairing and constructing neural-network weights through direct reasoning and tensor edits, without training.

Based on [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) by Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab (MIT CSAIL, November 2026), with links to their released experiment logs and exact Qwen edit. This is an independent implementation, not an official release from the authors.

The transformer-construction reference also draws on [Chandan Singh's toy-task report](https://csinva.io/weight-space-programming-autoresearch/analyze/toy_examples/report.html), including its lessons about evaluation leakage, fitting, architecture changes and misleading parameter counts.

## What it does

- Guides weight-only interpretation, bounded diagnostic probes and small-network construction.
- Turns a concrete failure into a localized, reversible tensor intervention with baseline comparisons and held-out evaluation.
- Provides CPU helpers to inspect safetensors, compare stored values and remove a direction from selected output-matrix columns.
- Documents exact-revision reproduction of the authors' Qwen3.5-0.8B loop correction.
- Explains algorithmic transformer circuits and provides independent parity/counting and position-routing examples with exhaustive finite-domain tests.

The checkpoint helpers do not discover behavioral directions or run inference. The toy examples execute their own small networks. The skill does not guarantee model improvements. Its tests validate tensor operations, artifact integrity and the toy circuits' bounded domains; the articles' benchmark gains have not been independently reproduced here. The original study also reports unsuccessful weight-only sycophancy inference and no established math gain for non-looping Gemma 3 270M.

## Install for Codex

Clone this repository as a skill folder:

```bash
git clone https://github.com/xero00000/weight-space-programming.git \
  "${CODEX_HOME:-$HOME/.codex}/skills/weight-space-programming"
```

Use the new skill in a fresh session. Other agents that support `SKILL.md` can use the same folder in their own skill directory. Agent-specific discovery and tool access depend on the host.

Example requests:

```text
Use $weight-space-programming to inspect this checkpoint's MLP weights.
Use only weight-space evidence; do not run the model.
```

```text
Use $weight-space-programming to investigate repeated thinking in this model.
Use at most 20 paired development prompts, preserve the original weights,
and propose a candidate only if the measurements support a localized fix.
```

```text
Use $weight-space-programming to encode this binary decision policy in a
specified small ReLU network without training. Explain the unit assignments.
```

## Use the helpers

The skill instructions themselves require no Python package installation. The optional scripts require PyTorch, safetensors and NumPy:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/weights.py inspect /path/to/model.safetensors
.venv/bin/python scripts/weights.py diff /path/to/base.safetensors /path/to/candidate.safetensors
```

To remove an evidence-derived direction from selected columns of an `[out, in]` matrix:

```bash
.venv/bin/python scripts/weights.py project /path/to/base.safetensors \
  --tensor model.layers.3.mlp.down_proj.weight \
  --direction /path/to/direction.json --columns 4,17,29 --alpha 0.5
```

This previews the edit without writing. Add `--output /path/to/new-candidate.safetensors` to create a checkpoint and hash manifest. The direction must be a nonzero JSON vector with one entry per output dimension. The values above are illustrative, not a proposed edit for a real model. Read [helper conventions and limits](references/tools.md) before writing.

Each command handles one file or shard. Projection preserves the original file and other tensors, computes in float64 and rounds back to the original dtype. Writing loads the full shard into RAM. The script does not assemble complete model directories, support quantized weight surgery, or apply the exact released Qwen patch. For that patch, use the [revision-pinned reproduction guide](references/qwen-reproduction.md).

## Try the construction examples

```bash
.venv/bin/python examples/transformer_circuits.py parity --bits 101101
.venv/bin/python examples/transformer_circuits.py route --tokens 3210 --position 1
```

The expected predictions are 0 (even parity) and 2 (the token at zero-based position 1). The networks use closed-form weights, causal attention and identity normalization; the router has no MLP. Their architecture changes are explicit. No training or pretrained weights are used. Summaries count frozen parameters and registered buffers, with source constants documented separately. See the [construction reference](references/transformer-construction.md) for equations and domain limits.

## Contents

| File | Purpose |
| --- | --- |
| [SKILL.md](SKILL.md) | Agent entry point and evidence/experiment workflow |
| [references/methods.md](references/methods.md) | Interpretation, targeted repair and construction methods |
| [references/transformer-construction.md](references/transformer-construction.md) | Circuit recipes, architecture constraints and honest evaluation/size accounting |
| [references/qwen-reproduction.md](references/qwen-reproduction.md) | Exact released artifacts, model revision and evaluation protocol |
| [references/tools.md](references/tools.md) | Commands, projection equation, memory and sharding limits |
| [references/sources.md](references/sources.md) | Research attribution and limitations |
| [scripts/weights.py](scripts/weights.py) | Deterministic checkpoint inspection, comparison and projection |
| [examples/transformer_circuits.py](examples/transformer_circuits.py) | Closed-form parity/counting and position-routing toy networks |

## Validation

```bash
python -m unittest discover -s tests -v
```

Tests check projection geometry, source preservation, unchanged columns/tensors, metadata, hashes, dtype rounding, invalid inputs and read-only preview behavior. Construction tests execute the toy networks on all 2,046 bit strings of length 1..10 and all 256 four-token inputs at each of four routing positions, in float32 and float64. They require no GPU, model download or large-model inference. Exhaustive finite-domain coverage does not establish generalization beyond those domains.

Original code and skill instructions are MIT licensed. Referenced models, logs, datasets and upstream patches retain their own licenses; see [sources](references/sources.md).
