# Checkpoint helper

Run `scripts/weights.py` relative to this skill's directory, using an environment with `torch` and `safetensors` installed. `requirements.txt` declares the dependencies; choose the PyTorch build appropriate to your system. The utility is CPU-only and never runs a model.

## Inspect and compare

```bash
python scripts/weights.py inspect /path/to/model.safetensors
python scripts/weights.py inspect /path/to/shard.safetensors --match mlp.down_proj
python scripts/weights.py diff /path/to/base.safetensors /path/to/candidate.safetensors
```

Inspection returns the file SHA-256, metadata, tensor shapes/dtypes, stored element count, nonfinite counts and finite-value norms. A substring filter limits tensor details while keeping the full file denominator. Inspection loads one tensor at a time. Diff loads a pair of tensors at a time and requires identical keys, shapes and dtypes; it rejects nonfinite and nonstandard floating tensors. F16, BF16, F32 and F64 are supported for diff/projection. Integer/quantized buffers require format-specific tooling. Numeric equality is exact within the stored dtype; signed-zero bit differences do not count as numerical changes.

## Preview or write a projection

Provide an actual evidence-derived direction as a JSON array. Its length must equal the **output** dimension of an `[out, in]` matrix. Explicitly choose input columns, which correspond to intermediate neurons for an MLP down projection. A toy direction might be `[1.0, 0.0, 0.0]` for a three-dimensional output; it has no general behavioral meaning.

```bash
python scripts/weights.py project /path/to/base.safetensors \
  --tensor model.layers.3.mlp.down_proj.weight \
  --direction /path/to/direction.json --columns 4,17,29 --alpha 0.5
```

Without `--output`, this computes the candidate and manifest in memory and writes nothing. Add a fresh path to save it:

```bash
python scripts/weights.py project /path/to/base.safetensors \
  --tensor model.layers.3.mlp.down_proj.weight \
  --direction /path/to/direction.json --columns 4,17,29 --alpha 0.5 \
  --output /path/to/candidate.safetensors
```

The output retains all other tensors and metadata. Its adjacent `.manifest.json` records the equation's parameters, direction, column indices, shape/dtype, requested versus changed counts, delta norms and source/output hashes. Check it before using the candidate. The projection normalizes the direction, computes in float64, and casts selected values back to their original dtype. Strength is restricted to `(0, 1]`. This helper intentionally exposes one operation; it is not an implementation of every edit in the article.

A write loads the whole input file/shard and needs extra memory for the modified tensor and serialization. Use an appropriately sized shard rather than attempting to rewrite a huge checkpoint on a small machine. Outputs and manifests must not already exist. The original is never overwritten. Files are staged in the destination filesystem and published without clobbering existing paths. A process or machine crash between publishing the checkpoint and manifest may leave a checkpoint without its sidecar; treat it as incomplete and regenerate under a fresh name.

For sharded Hugging Face models, copy the configuration, tokenizer, chat template, index and unchanged shards to a separate candidate directory, replacing only edited shards **under their original shard filenames**. Update the index if names or placement change. Unchanged shards can be copied or read-only linked, but do not mutate hard-linked originals. The helper does not assemble that directory. Sum counts over all model shards to obtain a full-checkpoint fraction; consider tied weights explicitly before presenting a unique-parameter count.

Restore by loading the original model directory. This utility does not modify runtime configuration or activate a candidate automatically.
