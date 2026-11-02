# Construct algorithmic transformers

Use this reference when writing weights for a known bounded algorithm. It adapts techniques and evaluation lessons from [Chandan Singh's toy-task report](https://csinva.io/weight-space-programming-autoresearch/analyze/toy_examples/report.html); the executable examples here are independent implementations.

## Fix the computation contract

Record input alphabet, padding, delimiter, maximum length, answer format, causal visibility, and output-token selection. Define which architecture changes are allowed before choosing a circuit. A frozen standard transformer and a custom attention/ReLU network are different targets. Replacing LayerNorm with identity, deleting modules, increasing width or adding a custom sparse operation must be disclosed and must respect the requested target.

For each layer, allocate residual channels by name and record when they are read and written. Keep content, position, working values and final readout separate unless a justified lifetime analysis permits reuse. Residual additions can corrupt a copied value if the destination already contains unrelated content. Record ranges and numerical margins at each step, including softmax approximation and the stored dtype.

## Reusable circuits

### Position routing with QK and content movement with OV

Put token and position encodings in disjoint channels. Let keys read a position code and queries represent the desired position. One-hot position codes give a unique maximum dot product at that position; signed codes give a Hamming-similarity construction. Values read only token content, and the output projection writes a separate destination channel.

Remember the actual score is `Q @ K.T / sqrt(d_head)`, plus the causal/padding mask. Finite softmax is never an exact selector. With score gap `g` over `m - 1` competitors, the target's attention probability is `1 / (1 + (m - 1) * exp(-g))`. Choose a margin that preserves the desired decision under residual mixing, depth, dtype and maximum sequence length; do not assume infinite sharpness. A masked-out source cannot be routed to a query regardless of score.

### Uniform counting and weighted sums

Set Q and K to zero to obtain a uniform average over **visible, unmasked positions**. To recover a count, multiply by the actual denominator. Padding, delimiter tokens and growing autoregressive prefixes change that denominator. A fixed multiplier is valid only for a fixed visible length or a proven compensation circuit. One-hot values turn this into a histogram.

For positive place weights `a_i`, desired scores `log(a_i)` make softmax proportional to `a_i`. Account for score scaling and divide by the known normalization: the attention result is `sum(a_i * v_i) / sum(a_i)`, not the raw sum. Ensure unwanted positions are masked or explicitly included in the normalization. Variable lengths need a corresponding construction.

### ReLU integer tables

On a bounded integer domain, `ReLU(x - k + 1) - ReLU(x - k)` acts as the indicator `x >= k`; it is a ramp between integers. A table `f(0)..f(M)` can be written as linear interpolation using `x` and shifted ReLUs, with coefficients given by its first/second finite differences. Three shifted ReLUs, `ReLU(x-k+1) - 2*ReLU(x-k) + ReLU(x-k-1)`, isolate integer `k` on that domain.

These circuits are only as exact as the integer representation and threshold margins. Multi-input tables can grow combinatorially. A table of GCD values derived from the task definition is a valid bounded implementation, but it is not Euclid's algorithm and does not establish extrapolation. Deriving an arbitrary label table from examples is data fitting/memorization, even without gradients.

### Arithmetic and reuse across answer tokens

For addition in base `b`, define per-digit generate `g_i = [a_i + b_i >= b]` and propagate `p_i = [a_i + b_i == b - 1]`. With zero incoming carry, carry-out at position `i` is the OR of `g_j AND p_(j+1) AND ... AND p_i` over `j <= i`, indexing from least significant to most significant. One-hot digit-pair features and Boolean ReLU circuits can implement these predicates. Long multiplication combines digit products and bounded carry propagation. State exact digit order and range of intermediate values.

When producing multiple answer tokens, distinguish values computed from the original prompt from state carried by generated tokens. Execute each token through the model. Input formatting, legal decoding and grading may live in the harness; arithmetic, lookup, sorting or copying the answer may not be delegated to a Python oracle during inference.

### Normalization

LayerNorm couples channels through their mean and variance, so a circuit derived for raw residual values may fail with normalization. If removal is allowed, disclose identity normalization. Otherwise derive the normalized values and thresholds. Dominant anchor channels can stabilize variance, but their scale, numerical conditioning and effect on every task channel require analysis; they do not make normalization exactly identity.

## Select and measure honestly

Separate constructive constants derived from the task's rules from coefficients inferred from labels. Least-squares solves, logistic regression, counted label statistics and fitted lookup tables remain fitting regardless of whether an optimizer or backward pass is used. Contrastive activation analysis for diagnosing an existing model is a different mode; document its data and use the skill's repair procedure.

Keep the evaluator outside the editable construction. Use development data to diagnose candidates; freeze weights and architecture before fresh-seed or exhaustive evaluation. Repeated access to a fixed evaluation seed turns it into development feedback. If any evaluation labels affected the construction, label the result contaminated and use a new independent evaluation. A fresh random seed does not guarantee disjoint examples in a small finite domain; check overlap or explicitly report exhaustive coverage.

Use a declared attempt/compute budget. After correctness is established, simplify within a separate bounded budget while preserving the verified domain and architecture constraints. Snapshot each candidate's weights, architecture, mechanism description and evaluation conditions. Do not adopt an indefinite research loop from an upstream prompt.

Compare total parameter entries, nonzero entries, parameter dtypes, buffer entries/bytes, lookup table sizes, and algorithmic constants stored in code. Count frozen and `requires_grad=False` parameters. Distinguish stored tensor bytes from serialized file size and runtime workspace. Enumerate plain Python or nonpersistent state separately; parameter and buffer inventories cannot discover constants hidden in source. Report comparable architecture families and domain bounds. Zero trainable parameters does not mean zero stored information; fewer parameters does not by itself prove a clearer mechanism.

Report sampled accuracy, exhaustive finite-domain coverage, and a mathematical proof as separate types of evidence. For large domains, include boundary cases (all-zero/all-one inputs, repeated symbols, maximum carries, padding transitions) and fresh examples after freezing. Shorter lengths or extrapolation beyond the constructed domain require their own evidence.

## Executable examples

From the skill directory:

```bash
python examples/transformer_circuits.py parity --bits 101101
python examples/transformer_circuits.py route --tokens 3210 --position 1
python -m unittest discover -s tests -v
```

`ParityCircuit` has one causal attention block and a ReLU MLP, with identity normalization and no position embedding. Ten left-padded input slots plus a delimiter give exactly 11 visible positions at the readout. Attention writes the count `N`; the MLP writes `N + 2 * sum((-1)**k * ReLU(N-k), k=1..9)`, which equals parity on integer counts 0..10. The final linear readout selects token 0 or 1. This is a custom small transformer construction, not a patch for a pretrained model. The tests enumerate all **2,046 nonempty bit strings of length 1..10** in float32 and float64.

`PositionRouter` uses one head, explicit token/position/routed channel groups, identity normalization and no MLP. It returns one selected token from four input positions using a finite attention score gap of 16. The tests cover **all 256 four-token inputs at each of four source positions** in both dtypes, checking classification and the attention margin.

Both construct weights from equations, freeze all parameters, and compute answers inside ordinary tensor forward operations. Test oracles are used only for verification. Their JSON summaries include all registered parameters and buffers; architecture choices and scalar constants are visible in the source. These tests establish the stated finite-domain behavior in the tested dtypes, not general language-model capability or reproduction of the report's 959 programs. Do not execute examples during a weights-only or zero-forward-pass assignment unless the task permits it.
