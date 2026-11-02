# Reproducing the released Qwen loop correction

This is a path to the **authors' released edit**, not a reproduction of the original editor trajectory. The full agent harness is not released, and the edit/benchmark results have not been independently reproduced by this skill's maintainers.

## Pin the artifacts

Use upstream log commit `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`:

- [Experiment 8 protocol](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/README.md)
- [Hash-checking edit application script](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/code/apply_edit.py) and its adjacent `st_numpy.py` dependency
- [50-pass released patch directory](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/edit): `manifest.json` and `edit.npz`
- Base: [`Qwen/Qwen3.5-0.8B` revision `2fc06364715b967f1860aea9cf38778875588b17`](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17)

Download only the needed source and artifacts into a separate experiment folder. Inspect code and licenses there. Do not silently substitute a newer revision. The base's tensor names are `model.language_model.*`; renaming tensors is not a substitute for matching their contents.

Once the artifacts are available, the upstream script accepts:

```bash
python apply_edit.py /path/to/base/model.safetensors /path/to/p50c/edit /path/to/candidate.safetensors
```

Choose a fresh output: upstream's script can overwrite an existing file. It verifies original and reconstructed tensor hashes; stop if they differ. Assemble a separate model folder using the matching configuration/tokenizer/template. Do not replace the only copy of the source weights. The upstream script can also emit per-tensor `.npy` files, which are not themselves a complete loadable model.

## What the released edit changes

The two matrices are:

```text
model.language_model.layers.15.mlp.down_proj.weight
model.language_model.layers.19.mlp.down_proj.weight
```

Both have shape `[1024, 3584]`. The edit targets 64 intermediate-neuron columns per matrix: 131,072 scalar locations. After evaluation's BF16 rounding, the released manifest reports 65,260 changed entries in layer 15 and 64,928 in layer 19, totaling **130,188 stored values**. The article's approximate 0.016% refers to the touched locations versus about 0.8 billion parameters. Do not report that every touched value necessarily changed or use only these two matrices as the full-model denominator.

Use the released patch for exact reproduction. A normalized projection derived independently with the generic helper is a **new candidate**, even if it edits these layers. The exact indices and values cannot be recovered from “64 columns” alone.

## Keep evaluation comparable

The published protocol enables thinking and uses temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5 and at most 32,768 new tokens. MATH-500 averages four samples per problem; GSM8K and IFEval average two. Preserve the pinned chat template, prompt formatting and answer extraction. IFEval uses the official strict prompt-level checkers on the answer after `</think>`.

The paper reports MATH-500 7.9% → 48.1%, GSM8K 13.8% → 67.1%, IFEval 21.0% → 62.3%, and MATH-500 token-limit hits 93.0% → 33.1%. These are **source-reported** results, not a promise for a different revision, runtime, quantization, template or seed. A shorter context or smaller sample count is a changed protocol and must be labeled accordingly.

For fresh experiments use training/development splits to select edits. Preserve the benchmark as held-out evaluation. The Gemma 3 270M follow-up found no established math improvement, supporting the distinction between removing repetition and adding missing reasoning capability.
