# Methods by task

## Read weights

First recover the computation: layer order, activations, normalization, residual paths, head grouping, positional encoding, feature order, and matrix orientation. An `[out, in]` linear map computes `y = x @ W.T + b`. Transposing a matrix interpretation changes the proposed mechanism.

For a small spatial policy, reshape first-layer weights using the documented channel/grid ordering. Locate spatial concentration and compare action readouts. Trace multiple layers before claiming that an input feature controls an action. For a generative MLP, output columns can reveal spatial structure, but edge-like components do not prove that samples are outlines: the article's shape interpretation missed filled interiors.

For transformers, inspect head-specific Q/K/V/O blocks with the actual architecture's packing, rotary operations, grouped heads and norms in mind. Algebraic compositions such as OV or QK can motivate hypotheses about copying or positional relationships; they are not activation measurements. The article's character-transformer inference used intermediate weights without embeddings/readout. Its successful copying-head analysis does not imply arbitrary hidden traits are recoverable from raw weights. Report uncertainty and plausible alternatives.

## Repair a localized failure

1. **Observe:** with allowed probes, collect successful and failing cases from development data under matched conditions. For looping, distinguish repeated reasoning, token truncation, ordinary wrong answers and grading/extraction errors. Check the documented decoding protocol before assigning a cause to weights.
2. **Localize:** inspect relevant attention and MLP output maps. If activations are permitted, compare productive and repetitive contexts with similar topic and position; document layer, token positions, sample selection and the chosen direction estimator. Subtracting a mean difference is a candidate mechanism, not ground truth. Validate on separate practice examples.
3. **Intervene:** choose a reasoned intervention: selected-neuron projection removal, a localized head-output ablation, or an explicitly derived replacement subcircuit. Evaluate ablations separately to determine which component matters.
4. **Control:** where affordable, compare a norm-matched random direction, a different layer or neuron subset, and a small set of strengths. A result that vanishes under matched controls weakens the claimed mechanism. Track every probe against the budget.
5. **Freeze and evaluate:** select on development results, then evaluate once on held-out examples and report uncertainty. An edit with no established gain should remain rejected.

### Direction removal convention

For an MLP output matrix `W` of shape `[d_model, d_ff]`, column `j` writes the residual vector associated with intermediate neuron `j`. Given a nonzero residual-space direction `d`, define `u = d / ||d||` and a chosen column set `J`:

```text
W_new[:, J] = W[:, J] - alpha * u[:, None] * (u.T @ W[:, J])[None, :]
```

At `alpha = 1`, selected columns are orthogonal to `u` in exact arithmetic. This does not guarantee that the full network stops exhibiting a behavior: other writers, biases and nonlinear paths remain. At reduced precision, some requested edits round away. Keep untouched columns bit-identical. Direction estimation, column selection and strength are experiment-specific. The helper implements this equation; it does not implement the authors' exact Qwen edit.

Use a limited strength comparison when justified by the hypothesis. Avoid unbounded layer/direction sweeps or fitting a direction to a benchmark. With zero permitted probes, label the result as an unevaluated intervention and do not invent a baseline score.

## Construct small networks

For algorithmic transformer constructions, use [transformer-construction.md](transformer-construction.md). It covers attention routing/counting, ReLU tables, arithmetic circuits, architecture and size accounting, and bounded exhaustive examples. The guidance below also applies to small policies.

Start with the exact state encoding and action mapping. Sketch a compact deterministic strategy: detect features, compute relations or short-range propagation, score actions. Assign units/layers and track their ranges. Set all intended weights and biases explicitly rather than leaving random unexplained paths.

For binary features `x_i` in `{0,1}`, `ReLU(sum(x_i) - (k - 1))` implements an AND of `k` inputs. For tanh, choose margins that distinguish states without assuming that approximate outputs are exact Boolean values. Include sign, scale, saturation and numerical precision in the derivation. Preserve pass-through signals in ReLU networks with paired positive/negative channels if signed values are required.

For spatial policies, compute geometry such as relative position or projected landing location only if the architecture has enough capacity and the input contains the necessary state. For board games, a layer may mark liberties or propagate local group signals; fixed-depth propagation is limited and does not solve arbitrary global connectivity.

Check legal-move masking and sampled versus greedy actions. If testing is allowed, evaluate fixed cases and full episodes separately; if it is forbidden, provide the weights and derivation without claiming measured success. Save an ordinary loadable checkpoint and the exact mapping from algorithm components to tensors.
