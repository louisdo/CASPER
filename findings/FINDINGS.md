# CSpR Findings Log

Running log of investigation findings. Newest first. Times are local (CDT).

---

## 2026-06-28 10:18 CDT — `CSpRCombinedTrain`: lambda, sparsity, and the softplus parametrization

Context: investigating the combined-score training class (`CSpRCombinedTrain`), which trains on a
single `score = score_token + lambda * score_phrase` instead of the separate per-partition fields
used by `CSpRTrain`. Config under discussion: `train/conf/cspr_keep_surface+combined.yaml`.

### 1. `loss_field_names: [score]` trains ONLY the combined score
- The loss sums one contrastive component per entry in `loss_field_names`
  ([train/train.py](../train/train.py) `compute_loss`, ~L168-172). With `[score]` it trains exactly
  the combined `score` field; `score_token` / `score_phrase` receive **no gradient**.
- The model still *emits* per-partition `score_<p>` keys from `forward`
  ([train/cspr/model.py](../train/cspr/model.py) ~L274-275, kept by `_build_score_dict`), but they
  are **not logged** as loss components unless they're in `field_names`. So with `[score]` the log
  line shows `score`, `reg_q`, `reg_d`, `l0_*`, `flops_warmup`, `lambda_*` — NOT
  `score_token`/`score_phrase`. (Seeing those in a log => it's an older run or a different config,
  e.g. `cspr_keep_surface.yaml` which uses `[score_token, score_phrase]`.)

### 2. `lambda_phrase` IS trainable, but looks frozen at 0.25 — parametrization + LR + rounding
- `lambda_raw[p]` is a real `nn.Parameter` in a `ParameterDict` (`_CombinedLambda`,
  [train/cspr/model.py](../train/cspr/model.py) ~L199-201), included in `model.parameters()` and on
  the loss gradient path via `combine()` (`score = combiner.combine(...)`, ~L295). bf16 is autocast
  via `TrainingArguments` (NOT a weight hard-cast), so `lambda_raw` stays fp32 and unfrozen.
  See [[bf16-use-autocast-not-weight-cast]].
- It *looks* stuck at 0.25 because:
  - Effective lambda = `softplus(lambda_raw)`; at init `lambda_raw = log(expm1(0.25)) ≈ -1.2587`,
    where the softplus slope is only **0.22** (compressive).
  - LR = 2e-5 (with warmup) is tuned for the transformer, far too slow for a single scalar.
  - Logs are rounded to 4 dp (`round(value, 4)`, train.py ~L124).
  - Net: ~**226 consistent-gradient Adam steps** just to move the *displayed* value by 0.001.
    Over a few 50-step log lines early in warmup, it rounds back to 0.2500.
- To verify movement: bump log precision to 6dp, inspect a checkpoint's `lambda_raw` vs -1.2587,
  or run longer. To make it adapt meaningfully: give `lambda_raw` its own higher LR (param group).

### 3. Combined vs. separate training → much sparser phrase partition (observed, first ~5k steps)
- User observation: phrase activations under `CSpRCombinedTrain` are *much much much* sparser than
  under `CSpRTrain` for the first ~5k steps; inferenced phrase reps "look amazing".
- Mechanism (FLOPS reg is identical in both setups — same `lambda_d/lambda_q`, applied to raw reps
  before any lambda weighting, train.py ~L178-180). The asymmetry is the **contrastive pull**:
  - `CSpRTrain`: `score_phrase` is a full standalone contrastive loss → strong "keep phrase active"
    pressure → denser phrase rep.
  - `CSpRCombinedTrain`: phrase enters the loss attenuated by λ≈0.25 and competes with the token
    partition (weight 1.0) which carries most of the ranking. Weak "keep active" pull vs. (eventually)
    full-strength FLOPS push → phrase collapses to only the truly-necessary dims → extreme sparsity.
- IMPORTANT caveat — the 5k-step timing: FLOPS uses quadratic warmup over `flops_warmup_steps=50000`,
  so at 5k steps FLOPS weight is only ~(5000/50000)^2 = **1%** of target. So the *early* sparsity is
  driven mainly by **redundancy with the token partition under the combined loss**, NOT FLOPS yet —
  FLOPS will deepen it post-warmup.
- Good or bad? Plausibly good (sparse = high-precision, interpretable, cheap to index). Risk =
  under-coverage / recall loss if FLOPS over-prunes concepts the weak ×0.25 signal never defended.
  Also: λ is ~frozen at 0.25, so the phrase partition is pruned while permanently down-weighted —
  it never gets to "earn" more weight. Connects to finding #2.
- To diagnose: track `l0_d_phrase` (already logged, train.py ~L191-195) and NDCG/recall side-by-side
  vs `CSpRTrain`; watch whether `l0_d_phrase` craters after FLOPS warmup (>50k).

### 4. How to choose `init_lambda` (discussion)
- `init_lambda` sets the starting ratio of phrase-signal to token-signal in the combined score
  (token fixed at 1.0). Because λ is ~frozen (finding #2), the init acts as a **near-permanent prior /
  hyperparameter**, not just a seed — and it decides the phrase partition's collapse-vs-populate fate
  during the critical early steps. It also implicitly gates the FLOPS tug-of-war (finding #3).
- Scale subtlety: `score_token` is **max**-pooled, `score_phrase` is **sum**-pooled — different raw
  magnitudes. λ must absorb that scale gap on top of expressing semantic preference.
- Recommendation:
  1. Measure empirical scale-neutral `λ₀ = median(score_token)/median(score_phrase)` on a real batch
     with the pretrained checkpoint (can do this in `evaluation/cspr/test_model_gitig_.py`).
  2. Apply a semantic-preference multiplier: 0.25–0.5×λ₀ for token-led, ~1.0×λ₀ for co-equal.
  3. The historical `0.25` bundles scale + preference; only a good guess if current sum/max scales
     resemble whenever 0.25 was tuned.
  - Cleanest fix: make λ truly trainable (separate higher LR), then init at scale-neutral λ₀ and let
    data find the preference — turns init back into a real initialization.

### 5. Why softplus (not ReLU) for the lambda weight — user dislikes softplus
- `weight()` returns `softplus(lambda_raw)` ([train/cspr/model.py](../train/cspr/model.py) ~L203-206).
  Purpose: keep effective weight ≥ 0 while differentiable everywhere — `d/dx softplus = sigmoid > 0`
  for all finite x, and output is always strictly positive (phrase weight never gated to exactly 0).
- ReLU is the wrong fix: `max(0, x)` has a **flat left half** — for `lambda_raw ≤ 0`, both value AND
  gradient are 0, so λ gets **permanently stuck at 0** (dead unit, unrecoverable). This is a *likely*
  failure here, not hypothetical: the phrase weight is under downward pressure (weak ×0.25 pull +
  FLOPS), so it tends toward 0 — softplus survives the excursion, ReLU would die at the first crossing.
- User's likely real objection: softplus distorts the init (0.25 → raw -1.26, slope 0.22 → part of
  why λ feels frozen) and is compressive near small λ.
- Better-than-ReLU alternatives discussed:
  - **Plain unconstrained `λ = lambda_raw`** (no activation), init 0.25 — slope-1 honest gradient, init
    means exactly what it says, no dead zone. Allows negative λ (meaningful: phrase mismatch as
    penalty). **Recommended**, paired with the separate higher LR.
  - `λ = exp(lambda_raw)` — strictly positive, multiplicative updates, never dead; can grow fast.
  - `abs`/softabs — positive, kink at 0 but gradient ±1 (not 0), so no dead unit.

### Pending / suggested follow-ups
- [ ] Give `lambda_raw` its own higher LR via an optimizer param group (unfreeze λ in practice).
- [ ] Optionally replace softplus with plain unconstrained λ (user preference).
- [ ] Add scale-measurement snippet to `evaluation/cspr/test_model_gitig_.py` to compute λ₀.
- [ ] Side-by-side `l0_d_phrase` + NDCG: `CSpRTrain` vs `CSpRCombinedTrain`, watched past FLOPS warmup.

---

## 2026-06-28 (earlier session 5546a90e) — Loss-function & negative-sampling design discussion (origin of `CSpRCombinedTrain`)

This is the design discussion that *produced* the combined-score class. Recorded here because it's the
"why" behind everything in the 10:18 section above. (Prior-session transcript, not memory-distilled.)

### The starting unease (token vs. phrase negatives)
- Current `CSpRTrain` design: matryoshka loss supervises `score_token` and `score_phrase`
  **independently** (`loss_field_names: [score_token, score_phrase]`). Token trained with hard negatives;
  phrase trained with **random/in-batch only** (hard-neg zeroed via `neg_usage`). At eval, phrase is
  down-weighted by ~0.25 (the `*0.5` in cspr_utils, effective ≈0.25).
- User's discomfort: "why must phrase be trained on random negatives only?" A legacy run showed training
  phrase on hard negatives gave bad results.

### Why phrase-on-hard-negatives failed in the legacy run (resolved)
- The legacy hard-neg mining: hard neg = a random doc from the **reference list of the same citing
  paper**. By construction these are co-cited → **maximally topically similar** to the positive →
  pos and neg share keyphrases almost completely → phrase margin ≈ 0. Forcing phrase to separate them
  manufactures distinctions that don't exist at phrase level → degradation.
- KEY CORRECTION (user): "hard ≠ topically similar" in general. Counterexample: "my friend is used to
  machine learning" vs "my friend learning to use a machine" — lexically near-identical but topically
  distinct; a cross-encoder separates them via *keyphrase* difference. So phrase **should** receive that
  class of hard negative.
- Synthesis: the real axis is **what kind of distinction a negative requires**, not hard-vs-random:
  - lexical-overlap / **same-topic** negatives (citation-list kind) → token's job (phrase margin ≈ 0).
  - lexical-overlap / **different-topic** negatives (compositional) → phrase's job.
  - So the legacy failure was specific to *that mining strategy* producing only same-topic negatives —
    NOT evidence that phrase can't take any hard negative.

### Combined score + mixed negative list + self-routing (the central idea)
- Train on combined `score = score_token + λ·score_phrase` (λ learnable), with a **mixed negative list**
  containing both negative types. No need to hand-route negatives to partitions:
  **softmax-over-the-list self-routes per candidate.**
  - same-topic candidate: phrase fires ≈equally on it and the positive → cancels in softmax → token
    gets the gradient.
  - topically-distinct candidate: phrase fires differently → contributes to softmax shape → phrase
    gets the gradient.
- This is **objective-agnostic**: works for KL distillation AND plain contrastive, because both are
  softmax-over-list. The self-routing comes from the combined score + softmax, not from KL specifically.
- Caveat for contrastive specifically: one-hot target *forces* a margin against same-topic negatives
  that may be **false negatives** (a co-cited paper that's actually relevant). KL avoids this — teacher
  gives soft mass to the relevant "negative," no false penalty. (Motivates eventual KL distillation.)

### "Separate the training" — what that should and shouldn't mean
- **Yes**: separate *negative population / supervision* per partition (matches each partition's capacity).
- **No**: separate *encoder / forward pass / gradient flow*. Both partitions are slices of ONE MLM head
  over ONE shared encoder (`PartitionedMaskedLM`) — they are not two networks. Isolating gradients or
  alternating would pull the shared encoder in inconsistent directions. Forward pass must stay single.

### KL vs MarginMSE (for the planned distillation)
- MarginMSE is pairwise + **scale-anchored** (pins `s_pos − s_neg` to teacher's absolute margin units) —
  fights the matryoshka design, which relies on scale-free per-field losses.
- KL is listwise + **scale-free** (matches teacher's *distribution shape* over candidates, temperatures
  handle scale) → much more natural for this architecture; also gives false-negative robustness cleanly.
- Decision leaning: KL distillation over the combined score, with a science-strong cross-encoder teacher,
  teacher scores **precomputed offline** (zero training-time teacher memory).

### Open concern: relative (list) vs absolute (corpus) supervision for phrase
- KL/contrastive over a curated list only constrains phrase *relative to candidates in the list* — says
  nothing about phrase's absolute activation geometry against the millions of un-listed corpus docs that
  first-stage retrieval actually runs against. A relative signal can be satisfied while the absolute
  inverted-index geometry is degenerate. (User pushed back that concept-driven hard negs largely cover
  this; partially conceded — the residual risk is the open-world/absolute-geometry regime, not vocab
  coverage.) → reason to possibly keep a contrastive term on phrase. Not resolved.

### Negative-list sizing (memory)
- In-batch negs are free (already-encoded other-query positives, via `score_batch=True` matmul) but
  **coupled to batch composition** — topically homogeneous batches (likely w/ citation sampling) → weak
  topical separators → phrase starvation.
- Explicit random negs decouple this (reliable topical contrast every step) but must be encoded (not
  free). Layout discussed: 1 pos + 2 mined (one per type) + ~8 random ≈ 11 doc-encodes/query ≈ 3.7×
  current cost — lands back on the memory concern, now self-inflicted. Trade-off left open.

### Paper / strategy context (non-code)
- This repo (CSpR) is the user's refinement of **CASPER** (arXiv 2508.13394, first author Lam Thanh Do —
  the user). It's an arXiv preprint not yet in a conference, so these improvements go *into the CASPER
  submission*, not a separate sequel — no self-plagiarism boundary.
- The typed-negatives-per-**granularity** idea is already in CASPER (so not fresh novelty), BUT it's
  novel *to the field* (neighbors like MGH/M3 differ negatives by *difficulty* or unify dense/sparse
  *functions* — a different axis).
- Goal: make CSpR **competitive with / top current embeddings on scientific retrieval specifically**
  (a winnable domain-specialist claim, not a frontier claim). Highest-leverage move per the discussion:
  **KL distillation from a science-strong cross-encoder teacher**, evaluated on scientific benchmarks.
- Secondary moat: the model is **interpretable** → can tag keywords to papers, not just retrieve.
- "Looks better (representation) but not yet more performant" is the current risky spot: reviewers
  discount representation claims and gate on metrics. The combined-score / distillation work is a
  performance play, which is the right axis.

### Implementation that came out of it (now in code, verified)
- `CSpRCombinedTrain` / `CSpRCombined` / `_CombinedLambda` added to model.py; `train_class` registry +
  `init_lambda` plumbing, λ excluded from weight decay, logged, persisted to `combiner_lambdas.json`;
  eval `init_model` builds `CSpRCombined` + loads λ + skips the `*0.5` hack.
- **√λ INVARIANT (subtle bug caught by user, fixed):** training `combine()` multiplies the *finished
  scalar score* by **λ** (applied once). Inference `scale_reps()` folds the weight into *both* the query
  and doc phrase reps *before* the dot product, which **squares** it → must use **√λ** so `(√λ)² = λ`.
  The original code used λ in both places → silently weighted phrase by λ² (0.0625 instead of 0.25) at
  eval, which would have made the combined experiment look broken. Invariant to protect: "both q and d
  encode through the same `scale_reps` exactly once." See [[combined-score-sqrt-lambda-invariant]].

---

## Earlier sessions (carried over from memory)

### 2026-06-27 — `keep_surface` experiment
- CSpR keyphrase rewrite axes: `mode` (append vs. inplace) × `keep_surface` (inplace only).
  `keep_surface=True` *inserts* the keyphrase token next to the surface phrase
  ("machine learning <<machine_learning>>") instead of replacing it.
- Finding: flipping `keep_surface: false → true` at **EVAL ONLY** on the inplace S2ORC model
  (pretrained with keep_surface=False) increased NDCG on 2 benchmarks (likely not universal).
- Why: replacing surface words strips lexical signal from the `token` partition (hurts ranking);
  keep_surface restores it while still feeding the keyphrase token to the `phrase` partition. Eval-only
  flip is handicapped — phrase embeddings are still OOD (trained without surface words).
- Hypothesis to test: full pretrain+finetune with keep_surface=True aligns phrase embeddings too,
  should match/beat eval-only-flip. Meaningful comparison = full-pipeline vs eval-only-flip number.
- Invariant: mode + keep_surface MUST match across all 3 stages (pretrain → finetune → eval/index),
  else broken phrase reps. Set up: pretraining-base.sh keep_surface block; train/conf/cspr_keep_surface.yaml;
  evaluation/cspr/conf/cspr_keep_surface.yaml.
- See [[keep-surface-experiment]].

### ~2026-06-24 — bf16: use autocast, never hard-cast weights
- For CSpR/SPLADE mixed precision, enable ONLY via `TrainingArguments(bf16=True)` (autocast + fp32
  master weights). Do NOT also load the model with `torch_dtype=torch.bfloat16`.
- Why: autocast runs matmuls in bf16 but promotes reductions (sum/log/softmax/FLOPS/loss) back to
  fp32. Hard-casting weights defeats this → two observed regressions: (1) phrase reps degrade (phrase
  partition is sum-pooled; bf16 accumulation loses rare small keyphrase activations); (2) token reps
  less sparse (tiny FLOPS-penalty gradients underflow in bf16 with no fp32 master copy).
- Apply: keep `BaseSparse` at float32 always; pass bf16/fp16 only to `TrainingArguments`. Health check:
  watch `l0_*_token` (should match fp32 run). If token L0 stays high even with fp32 loading, suspect
  `max_grad_norm` clipping the FLOPS gradient.
- See [[bf16-use-autocast-not-weight-cast]].

### ~2026-06-24 — partition gather → slice speedup
- In `PartitionedMaskedLM`, vocab partitions (`token`=0..30522, `phrase`=30522..len(tokenizer)) are
  contiguous. Selecting with a LongTensor index is advanced indexing (gathers/copies the full
  [B,S,~59419] logits ×2 partitions ×3 forwards/step). Replaced with **slicing** `[:, :, start:stop]`
  (a view: no allocation, no kernel) → large speedup even at 100% GPU util; precision-neutral.
  `__init__` detects contiguous (start, stop) once, falls back to index_select otherwise.
- See [[partition-gather-to-slice-speedup]].
