# Evidence for the TMR paper on DiscoForcing

Scripts that test the paper's analysis on the DiscoForcing continual run (AIST++ `aistpp_genre5`) and measure what
forgetting *is* in an audio-conditioned generator. Everything runs on the existing **vanilla** session checkpoints.
Buffer comparisons are training-free (all rules select at the same θ_{k−1}), so no TMR training run is needed.
`consolidate.py` writes one file to work from: `results/evidence/<benchmark>/EVIDENCE.md`. Each claim in it carries
its numbers, a verdict computed from them, the figure, and the place in the paper.

```bash
R=results/aistpp_genre5/vanilla bash scripts/evidence/run_all.sh
```

`R` is the directory your `scripts/eval_cl_session.py --out` JSONs (+ `.joints.npz`) go to. Re-run after S3 and S4:
φ and per-checkpoint losses are cached, and new boundaries and sessions are picked up automatically.

## What is tested, and how

| Paper statement | Measured quantity | Script → key in `EVIDENCE.md` |
|---|---|---|
| Thm 1: the top-M̂ eigenspace of C_k minimises Ψ among rank-M̂ projectors | Ψ(P̂) vs discarded mass; Ψ of random rank-M̂ subspaces and of random subspaces inside the top 2M̂ modes | e1 → A1 |
| BIC rank (Eq. bic) | M̂ with N_eff = ND and with N_eff = N, spectral gap, r_eff | e1 → A1 |
| Thm 2: ‖C_k − C_k*‖ = O(√(r_eff log T / N)) | subsampling curve against √(1/N′−1/N) (slope 1); held-out test windows vs training windows (data-reuse caveat of Sec. app:setup) | e1 → A2 |
| Prop. fw: OMP is fully-corrective FW, ε_mod ≤ 2 diam_Z/√(t+2) | ε_mod(t) per iteration vs the bound; fitted rate | e1 → A3 |
| Lemma modal | ε_B against √λ₁ ε_mod + tail, both halves of the proof separately, TMR and random buffers | e1 → A3 |
| Prop. 1: excess forgetting ≤ ηλ‖ḡ‖δ + Cη² | δ for every selection rule (unbiased, noise-averaged gradients), κ = δ/ε_B | e2 → A4 |
| "the analysis is symmetric in the temporal and channel axes" (Sec. scope) | which selection space's error rank-correlates with δ, and how stable κ is there | e2 → A5 |
| Eq. 4 / Eq. replay_forget / Prop. 1 / Cor. 1 in their own regime | one exact gradient step from θ_{k−1}: measured vs first-order change, bound, O(η²) remainder slope | e3 → A6 |

## New findings this adds to the paper

These are measurements, not assumptions. `EVIDENCE.md` reports whichever way each one comes out.

1. **Forgetting = lost audio grounding or prior drift? (B1).** The model is CFG-trained, so the old-session loss splits *exactly*: ΔL^cond = ΔL^null − ΔG, with G = L^null − L^cond the loss the model saves by listening to the music. A large grounding share −ΔG/ΔL^cond means the old genre's motion is still reachable but the old music no longer steers the model there. That is the empirical case for selecting replay on audio-encoder trajectories (A1). It belongs next to Fig. 1.
2. **Recency attractor and collapse of style information (C1).** For old-session music, the share of windows danced in the *newest* session's genre (vs the encoder's confusion on real dance), and I(music session; danced style) in bits. When this information goes to 0, the generator dances its latest style to every song. This is the generative analogue of task-recency bias in class-incremental learning, and it is new for motion generation.
3. **Rhythm survives, style does not (C2).** Beat alignment (repo BAS) of old-session dances against real dance, next to StyleRetention. If timing is kept and genre lost, forgetting is hierarchical. It matches Eq. 4: what all sessions share has aligned gradients, what is session-specific conflicts (B5).
4. **Where forgetting lives (B3).** The exact path integral of ∇L_j along the stored checkpoint path assigns every old session's loss increase to blocks: audio encoder, cross-attention K/V (which read the audio), Q/O, self-attention, FFN, modulation. This is compared with each block's parameter share and movement. Completeness is checked: the integral must reproduce the measured ΔL (adaptive Simpson, error reported).
5. **Diffusion noise limits gradient-space selection (A4).** The per-window gradient signal-to-noise for one diffusion draw shows how many noise draws gradient-matching coresets (GCR, GradMatch, the oracle here) need per sample. TMR's φ_i does not depend on the diffusion noise at all, which adds a variance argument to the paper's cost argument (Sec. scope (i)).
6. **What temporal modes are (B6).** Is C_k near-Toeplitz with DCT-like eigenvectors? If so, temporal modes are frequency bands of the audio-conditioned dynamics: mode 1 is the window average (static song/genre content) and the rest are within-window dynamics. The modal projector is then a data-adaptive low-pass filter and whitening equalises the retained bands; the "whitening gain" quantifies "balancing dominant and weak modes". Also reported: tempo and genre decodability on unseen songs, and subspace stability across checkpoints.
7. **When, and at which noise levels (B4, B2).** Is forgetting front-loaded in a session (a stability-gap overshoot)? At which diffusion-forcing noise levels does the old-session loss rise?
8. **Theory ↔ reported metrics (D1), and forecasting (D2).** Does the training-loss forgetting the analysis bounds track StyleRetention and FID_k forgetting? Do the first-order interference −⟨g_j, g_k⟩ and −cos at θ_{k−1} predict which old session the next session will hurt most, compared with the genre distance used to design the session order?

## A fix the paper needs (found while building this)

Algorithm 1 keeps only atoms with ⟨z_i, u⟩ > 0. That is the test for unconstrained or non-negative pursuit. On the
simplex (w ≥ 0, Σw = 1), an atom lowers ‖y − Zw‖ iff **⟨z_i − Z_S w_S, u⟩ > 0** (the Frank–Wolfe gap; by KKT,
⟨z_j, u⟩ = ⟨Z_S w_S, u⟩ on the support). The atoms are uncentred whitened trajectories sharing a large common
component, so after the first pick every raw correlation can be negative, and the literal rule returns a one-atom
buffer. It did on the synthetic test. Both rules choose the same atom whenever they continue, so Prop. fw (i)
stands, but Algorithm 1's 𝒬 and the remark inside Prop. fw should use the FW gap. `tmr_select.omp_simplex` uses
the FW gap by default (`stop="positive"` reproduces the literal rule), and E1 reports how many atoms the literal
rule selects on real data (A3). If your server implementation uses the literal filter, check its buffer sizes.

## Files

| file | role |
|---|---|
| `ev_common.py` | windows of tokens + token-aligned music (exact `AistPPDataset` alignment and length filters); frozen-noise diffusion-forcing loss (= `DiffForcingWanModel.forward` in eval mode with explicit noise); φ from `WanModel.music_embedding`; EMA checkpoint loading; per-block gradient tools; CountSketch; fp32 attention |
| `tmr_select.py` | Algorithm 1 (BIC rank, whitening, simplex-constrained OMP with exact KKT refit), Tab. 4 ablations, herding, k-center, ER, loss-based, gradient-sketch oracle |
| `e1_spectral.py` | A1–A3, B6, B7 (forward passes only) |
| `e2_gradients.py` | A4, A5, B5 (per-sample gradients, sketched) |
| `e3_probe_step.py` | A6 (exact full gradients, needs e2) |
| `e4_trajectory.py` | B1–B4 (every stored checkpoint of every session) |
| `e5_behavior.py` | C1, C2 (rescores the eval joints; no regeneration) |
| `consolidate.py` | D1, D2 and `EVIDENCE.md`, `evidence_tables.tex`, `evidence_summary.json` |

## Running on the server

- **Inputs.** Run dirs `outputs/<time>_cl_ldf_aistpp_genre5_s{k}/` with `last.ckpt` and the `step_step=*.ckpt` files, `raw_data/cl_sessions/aistpp_genre5` (manifest, lists, `_style_encoder/`), tokens and music features of the frozen VAE, and the wavs (tempo and BAS).
- **Settings.** EMA weights, fp32 everywhere (TF32 off, fp32 attention in place of the bf16 kernels, because loss differences of 1e-4 are measured). Windows are non-overlapping crops of `data.train_max_tokens` = 76 tokens (~10 s). That is the training window and the clip length of the replay buffers built by `scripts/build_replay_buffer.py`, whose `tmr` method calls `tmr_select.py`. `--max_windows N` caps windows per session if a pass is slow.
- **Rough A100 cost.** E1 ≈ 5 min per boundary. E2 ≈ 10–20 min per boundary (with `--R_eval 32`). E3 ≈ 10–15 min per boundary (if GPU memory is short, `--accum_device cpu`). E4 ≈ 1 h for four sessions (`--no_attr` skips the attribution). E5 a few minutes.
- **`eval_cl_session.py` change.** It now also stores the test sequence names, which E5 uses for BAS. Older JSONs work too: E5 reconstructs the names and checks the count.
- **Full m_{i,t} matrix.** Fill it with the optional loop at the end of `run_all.sh`.

## Caveats stated in `EVIDENCE.md`

- **Diffusion noise in gradients.** Per-sample gradients of a diffusion loss are noisy. δ is estimated without bias from two independent halves of the noise draws, and its Monte-Carlo share is printed. If that share is large, raise `--R_eval` or read A4/A5 as rankings.
- **Probe-step scope.** E3 tests one GD step, which is the update the analysis is about. It does not test Adam training over 40k steps; E4 covers that regime descriptively.
- **Data reuse.** θ_{k−1} was trained on the pool. A2's held-out check measures how much that matters, as Sec. app:setup asks.
- **Single vanilla trajectory.** Comparisons of rules are made at vanilla checkpoints. At k > 1 the TMR run's own θ_{k−1} would differ.
