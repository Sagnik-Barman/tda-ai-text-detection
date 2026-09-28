# The maths behind the project (interview prep)

## 1. Persistent homology in one page

Take a finite point cloud X with distances d(x, y). For each scale ε ≥ 0, the **Vietoris–Rips complex**
VR_ε(X) contains a simplex for every subset of points that are pairwise within ε. As ε grows the
complexes are nested — a **filtration** — so topological features appear ("birth") and disappear
("death"):

- **H0** — connected components. All points are born at 0; components die when they merge.
- **H1** — loops. A loop is born when a cycle closes and dies when it gets filled in by triangles.

Each feature gives a pair (birth, death); the multiset of pairs is the **persistence diagram** (or
barcode). Long bars = real structure; short bars = noise.

**Stability theorem** (Cohen-Steiner–Edelsbrunner–Harer 2007; Chazal et al. 2009 for Rips):
    d_B(Dgm(X), Dgm(Y)) ≤ 2 · d_GH(X, Y)
— perturb the points a little and the diagram moves only a little (bottleneck distance). This is why
topological features are robust to noise, and why they're a plausible candidate for robustness to
paraphrasing.

## 2. H0 persistence = minimum spanning tree (the key fact)

At scale ε the components of VR_ε(X) are the components of the graph with edges {d(x, y) ≤ ε}.
Sweep ε upward: two components merge exactly when the shortest edge between them appears. That is
**Kruskal's algorithm**: sort edges, add each one that joins two different trees (checked with a
union-find). So

    finite H0 death times  =  MST edge lengths.

`persistence.py` implements this three ways (union-find Kruskal, dense Prim, scipy) and the tests
check they agree with each other and with ripser.

**Threshold-graph Betti numbers for free.** For an attention graph with weights S, the graph
{S_ij ≥ τ} is the Rips graph of D = 1 − S at scale 1 − τ. Hence
    β0(τ) = n − #{MST edges of D with weight ≤ 1 − τ}
    β1(τ) = |E(τ)| − n + β0(τ)          (cyclomatic number of a graph)
One MST per head gives β0 at every threshold — no graph searches. A batched Prim over all 144 heads
makes attention topology ≈5.6× faster than per-head scipy MSTs (0.29 s vs 1.62 s per 256-token text).

## 3. Persistent-homology dimension (PHD)

For n i.i.d. points from a measure on a d-dimensional manifold, let E_α(X_n) = Σ_{MST edges} |e|^α.
The Beardwood–Halton–Hammersley / Steele scaling law says

    E_α(X_n) ≍ n^{(d − α)/d}      (0 < α < d).

Intuition: n points in a d-dimensional region have typical nearest-neighbour spacing n^{−1/d};
the MST has n − 1 edges of about that length, so E_α ≈ n · n^{−α/d}.

Taking logs, log E_α = c + m·log n with slope m = (d − α)/d, so

    d = α / (1 − m).

**Estimator** (`phd.py`, following GPTID): sample subsets of sizes n_1 < … < n_k, compute E_α for each
(median over several subsets), fit m by least squares, return α/(1 − m). The sanity script recovers
circle → 1.04, sphere → 2.03, swiss roll → 2.00, 5-D Gaussian in ℝ⁷⁶⁸ → 5.01.

**Why AI text might be lower-dimensional (hypothesis, not theorem):** LLM decoding concentrates
probability on high-likelihood continuations, so contextual embeddings of generated tokens vary
along fewer effective directions than human text, which mixes registers, idioms and surprises.
Tulchinskii et al. report human text at ≈9 (English, RoBERTa) and AI text about 1.5 lower. Known
failure modes: high-temperature sampling (flattens the distribution) and short texts (too few points).

**Why PHD rather than PCA?** PCA measures *linear* dimension; token clouds lie on curved manifolds.
PHD, MLE and TwoNN are *intrinsic* (local / multi-scale) estimators.

**Other estimators used as features:**
- MLE (Levina–Bickel): m̂(x) = [ (1/(k−1)) Σ_{j<k} log(T_k(x)/T_j(x)) ]^{−1}, averaged as an inverse mean.
- TwoNN (Facco et al.): μ = r₂/r₁ follows a Pareto law with −log(1 − F(μ)) = d · log μ; fit the slope.

## 4. Likelihood baselines

For a causal LM p and tokens x_1..x_T:
- **log-likelihood** (1/T) Σ log p(x_t | x_<t); **log-rank** (1/T) Σ log rank(x_t).
- **Fast-DetectGPT** (analytic): d = (Σ_t log p(x_t) − Σ_t μ_t) / sqrt(Σ_t σ²_t), with
  μ_t = E_{x̃∼q_t}[log p(x̃)], σ²_t = Var_{x̃∼q_t}[log p(x̃)]. Machine text sits above what the model
  would itself typically sample.
- **Binoculars**: B = log PPL(x) / X-PPL, where X-PPL is the cross-entropy between observer and
  performer next-token distributions. It normalises away "this topic is just predictable".

## 5. Evaluation choices

- **Grouped splits**: human and AI answers to the same question share a group and never straddle
  train/test — otherwise a model can learn the question, not the author.
- **TPR@1%FPR**: AUROC can look great while the operating point is useless; in practice you care
  how many AI texts you catch while flagging ≤1% of humans.
- **Robustness gap** = in-domain AUROC − mean out-of-distribution AUROC. The project's headline is
  which detector has the smallest gap, not which has the highest in-domain score.

## 6. Likely interview questions (short answers)

- *Explain persistent homology in 60 seconds.* → Section 1: grow balls, track when components and
  loops appear and die, long bars are signal.
- *Why is H0 persistence just an MST?* → Section 2 (Kruskal).
- *How does PHD work?* → MST length scales like n^{(d−α)/d}; fit the log-log slope.
- *Why would AI text have lower intrinsic dimension?* → decoding concentrates on likely continuations;
  it's an empirical finding and fails at high temperature — you measured where.
- *Complexity?* → PHD: O(k·s·n²) for k sizes × s subsets of up to n points (n ≤ 512); attention:
  144 heads × O(T²) with batched Prim. Full Rips H1 would be O(n³) — capped with subsampling.
- *Why not just fine-tune RoBERTa?* → it wins in-domain but typically loses more on unseen
  generators/domains; the grid quantifies that trade-off.
- *Would you deploy this?* → Not for punitive decisions: false positives on non-native writers,
  paraphrase attacks. Useful as one signal among several, with human review.
