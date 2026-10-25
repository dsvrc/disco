# DiscoForcing as a fourth TMR benchmark: data partitions, order, metrics and training plan

The goal is to add music-to-dance generation with DiscoForcing to the TMR paper as benchmark (iv), alongside GestureLSM/BEAT2, TalkSHOW and MOSPA/SAM. Session partitions are in `raw_data/cl_sessions/`, built by `scripts/make_cl_sessions.py` (deterministic, seed 0). Evaluation references are in `raw_data/cl_sessions/<benchmark>/eval/`, built by `metrics/cl_metrics.py`.

---

## 1. What this benchmark adds to the paper

| Gap in the current paper | What DiscoForcing adds |
|---|---|
| Three generator families (flow matching, AR VQ-VAE, diffusion) | A **fourth**: causal, streaming **latent diffusion forcing**. This directly tests the "architecture-agnostic" claim. |
| Audio is speech or spatial audio | **Music**, encoded by a causal VQ-PAE. |
| The behavioural-mode axis has 3 sessions (MOSPA); long-horizon evidence covers only speakers | **5-session and 10-session dance-genre benchmarks**, so the behaviour axis gets long-horizon results too. |
| Forgetting is shown with FID/FGD only | An **interpretable, validated style-retention number**: how much of old-genre music is still danced in the old genre, measured by a learned style encoder that ignores quality defects (StyleRetention, §4). |
| Only within-dataset shifts | **FineDance style families**: a cross-cultural shift. |

## 2. The chosen order: escalating drift (one order per benchmark)

**Criterion.** Under vanilla fine-tuning, forgetting of session i after training session t grows with how far S_t is from S_i (the gradient conflict of Eq. 6). The order is chosen so that **the distance from every old session to each new session rises monotonically**. Vanilla forgetting then compounds steadily, as in your Fig. 3b, instead of oscillating when a similar genre comes back (the pattern vanilla showed on MOSPA). Among such orders, the one with the largest predicted vanilla F_avg (your weighting) is chosen, with ties going to the larger S0. The search is exhaustive over all orders.

### Main table: `aistpp_genre5`
| Session | Genres | Train / val / test seqs | Train min | Distance from S0 |
|---|---|---|---|---|
| S0 | LA Hip-hop + Lock | 196 / 4 / 42 | 48.1 | – |
| S1 | Middle Hip-hop + Pop | 195 / 4 / 42 | 46.9 | 0.41 |
| S2 | Krump + Waack | 187 / 4 / 41 | 43.5 | 0.52 |
| S3 | Break + House | 194 / 4 / 35 | 41.5 | 0.86 |
| S4 | Ballet Jazz + Street Jazz | 180 / 4 / 40 | 43.0 | 1.34 |

- Every old session's curve is monotone (1.00), and it has the highest predicted F_avg of all 120 orders.
- S0 is the most coherent session (within-session distance 0.26) and the largest.
- The story escalates in three steps: drift within hip-hop (S1–S2), a motion-style jump to floor work (S3), then a change of the *music itself* to jazz/piano (S4; music distance about 7× larger). That last shift is the biggest change in audio-encoder trajectories, exactly where TMR's theory says replay of temporal modes matters most.
- The contrastive order used earlier (max consecutive distance) made S0's distance oscillate (0.86, 0.41, 1.34, 0.52), so it was dropped.

### Long horizon (analogue of Fig. 3b): `aistpp_genre10`
LO → LH → MH → KR → HO → JS → PO → WA → BR → JB, one genre per session (about 98 train / 2 val / 20 test each).
- S0's distance curve rises strictly (0.26 → 2.16), and overall monotonicity is 0.95.
- S0 (Lock) is part of the 5-session S0.

### Cross-cultural: `finedance_style4`
Mix (36 train seqs) → Street (66) → Folk (23) → Classic (41). S0's distance curve is 0.44 → 0.84 → 1.31 (monotone 1.00). Test = the official `cross_genre` test per family. Val = 10% of train per family.

**Shared AIST++ protocol.**
- Train = official `crossmodal_train_without_val` (ignore list removed).
- Test is **music-disjoint**: the official test plus the unused sequences on each genre's held-out music piece, about 20 per genre instead of 2.
- Val = official val.
- Each genre's 3 dancers appear in no other genre.
- Genres are paired by minimizing the worst, then the total, within-session distance (mean of motion and music Fréchet distances).

## 3. Where TMR plugs in (assumption A1)

- Music enters only through **`WanModel.music_embedding`** (MLP 35→1024→1024 on the causal librosa windows; 16→ with VQ-PAE; `models/tools/wan_model.py:411`), whose output is the cross-attention context.
- TMR's ϕ_i ∈ R^{T×1024} is that output mean-pooled over the causal window W, at the latent frame rate, on fixed-length clips (e.g. 240 frames = 60 latents = 8 s).
- `DiffForcingWanModel.music_embedding` (`models/diffusion_forcing_wan.py:67`) is unused. Don't hook it.
- Replay: 10% per past session, λ = 1, OMP (CoSaMP / SSP as variants).

## 4. Metrics: why the stock ones would hide forgetting, and the protocol to use

**What was wrong**, measured on ground truth alone. "Complete forgetting" means session i's GT is scored against another session's GT, i.e. a model dancing the wrong genre. That is compared with the split-half noise of session i's own GT.

| Metric | AIST++ signal-to-noise (min / median) | FineDance | Problem |
|---|---|---|---|
| FID_k, as in `metrics/m2d.py` | −0.6 / −0.2 | −1.3 / −0.9 | Kinetic-energy statistics barely differ between genres. One feature per sequence (~40 samples per session, 3–8 for FineDance) gives a rank-deficient 66×66 covariance. |
| FID_g, as in `metrics/m2d.py` | overflows (up to 10¹⁶) | overflows | Re-normalized per call by std + 1e-10, so any geometric feature constant in the reference explodes. |
| FID_k / FID_g, frozen pooled normalization | −0.7 / 0.9, −0.7 / 0.4 | negative | Stable, but still no signal. |
| GenreAcc (linear probe on the same handcrafted features) | 12.3 / 19.5 | 3.0 / 11.6 | Separates GT well (0.82–0.93 → 0.03–0.11), **but fails the validity tests below.** |

**Validity tests of the probe** (GT only, `aistpp_genre5`):

| Test | Result | Meaning |
|---|---|---|
| Dancers held out (each genre has 3 exclusive dancers) | genre accuracy 0.85 → **0.64** (chance 0.10; humans 38%, ridge 48% in the literature) | The genre signal is real, but about a quarter of the accuracy was dancer identity. |
| Tempo ×0.8 / ×1.25 | 0.86 → 0.83 / 0.86 | Robust. |
| Jitter 0.5 cm | 0.86 → 0.61 | Fragile. |
| **Temporal smoothing σ = 2 frames (0.07 s)** | **0.86 → 0.22**; almost every window becomes "Krump+Waack" | **Confuses quality with style.** Slightly smoother generations would read as forgetting. |

**Resolution: a learned style encoder, validated** (`metrics/style_encoder.py`; checkpoints and reports in `raw_data/cl_sessions/_style_encoder/`). No standard style metric exists in the dance literature (Robust Dancer uses an SVM on the handcrafted features above; MDSC uses a joint music-motion embedding), so the encoder ships with its own validation suite. It can be a small contribution of the paper.

- **Input:** global joint positions (what any generator's output recovers to), expressed relative to the pelvis and rotated to the dancer's facing direction, plus root height, root travel velocity and joint velocities. Exactly yaw-invariant.
- **Model:** a 1.2M-parameter temporal conv net over 5 s windows (1 s stride), trained with genre cross-entropy plus a supervised contrastive loss, then temperature-calibrated.
- **Training data:** the session train lists of all sessions, the same data the handcrafted probe used. It is an evaluator only and never touches the learner.
- **Augmentations** (invariance to generator artifacts): smoothing σ 0.5–4 frames, jitter 0.2–2 cm, tempo ×0.8–1.25, body scale ±10%, slow root drift. Mirroring, frame stutter, low-frequency wobble and strengths beyond these ranges are never used in training, so they test generalization.

**Validation on `aistpp_genre5`** (criteria fixed before running; figure: `_style_encoder/aistpp/validation.png`):

| Test | Learned encoder | Handcrafted probe | Criterion | Verdict |
|---|---|---|---|---|
| Clean, music-disjoint test windows | genre 0.92, session 0.94 (calibration error 0.07) | session 0.86 | – | – |
| Held-out dancers (10 genres; 3 folds) | **0.81** (0.81 / 0.81 / 0.82); 0.79–0.84 when dancer *and* music are unseen | 0.65 | beat the probe | **pass** |
| Seen corruptions, in range (smoothing, jitter, tempo) | worst drop **0.011** | worst drop 0.63 (jitter 2 cm: 0.23) | ≤ 0.05 | **pass** |
| Unseen or out-of-range (mirror, stutter, wobble, combined, σ = 6, 3 cm, ×0.7, ×1.4) | worst drop **0.103** (3 cm low-frequency wobble); every other case ≤ 0.05 | 0.18–0.64 | ≤ 0.10 | narrow miss |
| Genre swap (retention matrix) | right session 0.79–0.87; wrong session ≤ **0.08**; hard accuracy 0.87–0.96 vs ≤ 0.08 | – | ≥ 0.8 / ≤ 0.1 | narrow miss (Jazz session 0.79) |
| Signal-to-noise of detecting complete forgetting | **18–54** (style retention), 7–20 (StyleFD) | – | – | – |

The two near-misses are reported as measured rather than tuned away, which would mean fitting the test set. Meanwhile FID_k explodes under the same corruptions (2 cm jitter: 3,041) while the encoder holds 0.93: it separates quality from style. Drift between a dance and its own smoothed copy is 0.0001.

**Validation on `finedance_style4`** (4 families; FineDance has no dancer labels, so no held-out-dancer test; figure: `_style_encoder/finedance/validation.png`):
- **Clean:** 0.90 vs the probe's 0.79.
- **Seen corruptions:** worst drop 0.012 (probe: up to 0.42).
- **Unseen:** worst drop 0.102 (3 cm wobble), so the same narrow miss as AIST++.
- **Swap:** Folk 0.92 and Classic 0.93 vs ≤ 0.04 (signal-to-noise 146–577). **But Mix and Street overlap:** Mix's GT retention is 0.73, Street dance scores 0.21 as Mix, and Mix's noise is high (std 0.12, only 4 test sequences). The Mix family is mostly K-pop and choreography, which shares moves with street dance. Forgetting Mix after learning Street would be partly masked.
- **Recommended fix:** drop Mix, giving 3 sessions, Street → Folk → Classic. Distance from S0 is 1.10 → 1.64 (monotone), S0 is the largest family, and every pair separates crisply. Alternatively keep Mix and report its higher noise floor.

**Floor test on generated motion, and the reject class (default encoder since 2026-10-21).** The first encoder failed on real generator output.
- **The failure:** a randomly initialized generator, decoded by the frozen VAE, scored S0 StyleRetention **0.58** (real dance 0.88), and a step-2,500 S0 model scored the same (0.59).
- **The cause:** the softmax must put every window on some genre. Decoder garbage, such as random latents and the untrained generator, lands on **Lock** (an S0 genre: 67–75% of windows). Everyday motion lands on **Jazz** (S4 mass 0.52).
- **The fix:** an 11th class, `other`, trained on non-dance negatives: BABEL-272 everyday motion with dance-annotated sequences excluded (40%), random latents decoded by the frozen VAE (30%), static poses (15%), time-shuffled dance (15%). Code: `train(..., negatives=True)`; the old encoder is kept in `_style_encoder/aistpp_v1/`.

| Validation (`aistpp_genre5` test) | Old encoder | Reject-class encoder (default) |
|---|---|---|
| Real dance: window session accuracy / retention | 0.938 / 0.85 | 0.920 / **0.91** |
| Real dance identified per dance (majority vote) | 198/200, margin median 0.90 | 198/200, margin median **1.00** |
| Unseen dancers (10-genre accuracy) | 0.81 | 0.77 (probe 0.65) |
| Smoothing / jitter / tempo in trained range, worst drop | 0.011 | 0.018 |
| Genre swap: right vs wrong session | 0.79–0.87 vs ≤ 0.08 | **0.85–0.95 vs ≤ 0.08** |
| Non-dance probes (untrained generator, static, shuffled, random latents, everyday motion): mass on any session | up to 0.66 | **≤ 0.017** |
| Garbage never trained on (constant latent, latent random walk) | – | ≤ 0.12 |
| Worst unseen distortion | 0.10 (wobble) | **0.175** (3 cm low-frequency wobble) |

On generated motion (first 10 s of each test dance, S0 retention): untrained generator **0.017** < S0 model at step 2,500 **0.34** (57% of its windows still not dance-like) < tokenizer ceiling 0.90 < real dance 0.95. The same step-2,500 model on **Jazz (S4)** music scores S4 retention **0.002**: about 90% of its dance-like output is S0 style. It dances the one style it knows, which is exactly the signature forgetting produces.

**Known limitations, to state in the paper:**
- Near-static recordings carry no identifiable style. The reject class refuses them: 2 of 200 real test dances (mean joint speed 0.08 and 0.30 m/s vs a median of 0.72) and windows with mocap glitches, so 2.7% of real test windows in total.
- Low-frequency wobble (3 cm) partly reads as non-dance (retention 0.91 → 0.74). The fix is to add mild wobble (≤ 2 cm) and mirroring to the positive augmentations and retrain (about 40 min).
- Street Jazz alone is often confused with Ballet Jazz; as a single-genre session (`aistpp_genre10` S5) its retention is lower. In `aistpp_genre5` the two share a session.
- A small human 2AFC agreement study remains optional.

**Per-session metrics** (`CLEvaluator.session_metrics`, input = global joints of all generated samples for session-i music):
- **StyleRetention (primary):** the mean probability mass on session i's genres. Forgetting = m_{i,i} − m_{i,t}. Non-dance output scores near 0, so it can't masquerade as retention. Reference levels for each session are in `eval/noise.json`.
- **DanceRate:** 1 − P(other), the share of the output recognized as dance at all. It separates "forgot the style" (danced another session's genre) from "stopped producing dance".
- **StyleAcc:** the hard (arg-max) version.
- **StyleFD:** Fréchet distance of embeddings to session i's references (VAE reconstructions of its test dance).
- **drift(a, b):** label-free StyleFD between two generated sets, e.g. the model right after session i vs after session t.

**Protocol pieces that stay:**
- **Quality columns: FID_k, FID_g** on the same windows, against session i's reference, with **one frozen benchmark-wide normalization** (constant features dropped). Comparable with the literature, and they never blow up.
- **BAS: sanity check only.** Every genre is danced on the beat.
- **Sampling:** 5 s windows with a 1 s stride, plus 3 generated samples per test music, pooled.
- **Reference:** VAE-reconstructed GT (`fid_target: vae`), built with `python -m metrics.cl_metrics build --benchmark <b> --ref_feature_dir raw_data/aistpp/RECOVERED_cl_aistpp_genre5_alltrainvae`.
- **Sanity rows to report next to every checkpoint:** untrained generator (floor), VAE reconstruction (ceiling), real dance (`scripts/eval_cl_session.py --untrained / --vae_recon / --gt`).
- **Noise floors** and complete-forgetting levels per session are in `eval/noise.json`. Report them.
- **Summaries:** F_avg / A_avg **only** through `forgetting_summary()`, which implements your definitions exactly (verified against your Table 3).
- The handcrafted `GenreProbe` stays only as the validation baseline.

## 5. Frozen components and leakage

| Component | Trained on |
|---|---|
| Music features | **Fixed signal processing** (causal librosa windows), so nothing is learned and nothing can leak. If VQ-PAE replaces them, train it on GTZAN, an external dataset (`PAE_training/.../configs/vqpae_gtzan_1000eps_8emb.yaml`). |
| Motion VAE (tokenizer) + normalization | **Train split of all sessions, never test data**, pretrained once and frozen (`configs/cl/vae_aistpp_genre5_alltrain.yaml`, `alltrain_{Mean,Std}_272.npy`). All continual learning happens in the generator. |
| Session design | Train-set statistics only |

**Why the tokenizer can't be leakage-free (measured 2026-10-20).** A frozen VAE must reconstruct every genre, or later sessions can't be learned at all. With the style metric on reconstructions of each session's real test dance (real dance: 0.79–0.87):

| Tokenizer | S0 | S1 | S2 | S3 | S4 |
|---|---|---|---|---|---|
| S0 only (2.5k steps) | 0.90 | 0.25 | 0.27 | 0.15 | 0.09 |
| S0 (×10) + BABEL-272 general motion (2.5k steps) | 0.89 | 0.42 | 0.16 | 0.28 | 0.15 |

Both collapse unseen genres onto the training genres (51–76% of windows classified as LA Hip-hop / Lock). An untrained generator decoded by the S0-only VAE even scored S0 retention 0.86. So the paper should state: *the motion tokenizer is pretrained once on the training split of all sessions and frozen; continual learning applies to the generator, which is where TMR operates.* This is the standard frozen-VAE convention of latent generators.

## 6. Training stages (as built for `aistpp_genre5`; status as of 2026-10-20)

| Stage | What | Command / config | Status |
|---|---|---|---|
| A | **Music features:** causal librosa windows (64 frames × 35 dims: envelope, MFCC, chroma, peaks, beats). No learned audio front-end, so nothing can leak future sessions and nothing needs training. VQ-PAE on GTZAN stays an optional upgrade. | `python scripts/librosa_preprocess_parallel.py --wav_dir raw_data/aistpp/wavs --dest raw_data/aistpp/baseline_feats_window_64 --window_size 64 --lists raw_data/cl_sessions/aistpp_genre5/joint_{train,val,test}.txt` | done, 1,172 sequences |
| B | **Motion VAE pretrained on the train split of all sessions**, then frozen (S0-only and S0+BABEL failed, §5) | `python scripts/make_cl_vae_data.py --alltrain`, then `DISCO_CONV_GEMM=1 python train_vae.py --config configs/cl/vae_aistpp_genre5_alltrain.yaml --override train=True --override trainer.max_steps=10000` | training |
| B2 | **Tokenize all five sessions** with the frozen VAE (tokens + reconstructions) | `python pretokenize_vae.py --config configs/cl/vae_aistpp_genre5_alltrain.yaml --override test_ckpt=<vae ckpt>` | after B |
| C | **S0 diffusion from scratch** | `python train_ldf.py --config configs/cl/ldf_aistpp_genre5_s0.yaml --override train=True --override test_vae_ckpt=<vae ckpt>` | after B2 |
| C2 | **Score checkpoints** with the style metric (m_{0,0}) and score the VAE ceiling for every session | `python scripts/eval_cl_session.py --config configs/cl/ldf_aistpp_genre5_s0.yaml --vae_ckpt <vae> --ckpt <ldf> --session 0` (add `--vae_recon` for the ceiling) | after C |
| D | Sessions 1..K per method, warm start (fresh optimizer) + replay of 10 % of every past session's clips (`utils/cl_replay.py`, buffers by `scripts/build_replay_buffer.py`: random, ocs, gcr, tmr, tmr_mix, tmr_joint, tmr_joint_ess, tmr_joint_neff, tmr_joint_op, tmr_dance, tmgm, mir, csrel; STAR via `replay.star`; distillation replay via `replay.distill`) | `bash scripts/cl/run_<method>.sh` (vanilla, random, random_star, random_distill, ocs, gcr, tmr, tmr_mix, tmr_joint, tmr_joint_ess, tmr_joint_neff, tmr_joint_op, tmr_dance, tmr_dance_sel, tmgm, mir, csrel); diagnostics: `scripts/cl/audit_buffers.py`, `scripts/cl/diag_replay.sh`, `scripts/cl/check_tmr_duplicates.py`; S0->S1 screen: `scripts/cl/screen_s1.sh` (reuse a screen S1 run in the full chain: `scripts/cl/promote_screen_s1.sh`) | written, smoke-tested on fake data |
| E | Joint-training upper bound | `joint_train.txt` | later |

Methods, all with the same buffer and steps: Vanilla, TMR (OMP / CoSaMP / SSP), ER, MIR, GCR, CSReL, OCS (+ EWC / C-Flat / SD-LoRA if budget allows). Evaluate after every session on every seen session's test set to fill m_{i,t}.

**Compute on this machine (Radeon 8060S, ROCm).**
- MIOpen runs the VAE's 1-D convolutions through a slow fallback. `DISCO_CONV_GEMM=1` computes them as unfold + matmul instead (exact), giving about 1 s per step at batch 196 × 64.
- The style encoder is faster on the CPU (`STYLE_ENCODER_DEVICE=cpu`).
- Run the method sweep on the A100. Priority: `aistpp_genre5` (Vanilla, TMR, ER, joint), then the remaining baselines, then `aistpp_genre10`, then FineDance.

**Environment fixes found on the way:**
- numba is pinned to 0.62.1, because numba 0.68 segfaults inside librosa's beat tracker and silently hangs a multiprocessing extraction.
- Training lists must live in the dataset folder (`raw_data/aistpp/cl_<benchmark>_s{k}_{split}.txt`, now written by `make_cl_sessions.py`), because the loaders resolve paths relative to the list file.

## 7. Remaining work

1. Finish B → B2 → C above, then the VAE-domain check of the style metric on reconstructions of every session (`--vae_recon`).
2. Write the session loop for D: warm start from the previous session, replay buffer mixing, TMR buffer construction on the `WanModel.music_embedding` hook, and per-session evaluation via `scripts/eval_cl_session.py` + `forgetting_summary()`.
3. FineDance slicing for training (`utils/dataset_slice.py` + `split_new_dataset_from_txt`), and the Mix-family decision (§4).
4. Optional: VQ-PAE on GTZAN instead of librosa features (needs `audiotools` installed with `--no-deps`).

## 8. Fix in the paper before submission

Recomputing your summary rows from your own tables, with your own formulas:
- **Table 1 (BEAT2):** 12 of 13 F_avg values don't reproduce. Only TMR (0.185) does. Vanilla recomputes to **0.989** (reported 0.698); ER to 0.547 (reported 0.454); STAR and GenRep are reported *higher* than recomputed. TMR still ranks first.
- **Table 2 (TalkSHOW):** F_avg reproduces, but **every A_avg is the last-session (Chem) value** rather than the mean over new sessions used in Tables 1 and 3. For example, Vanilla should be 3.879, not 3.658.
- **Table 3 (MOSPA):** everything reproduces.
- **Abstract, "up to 75%":** TalkSHOW gives 1 − 0.346/5.048 = 93%, and corrected BEAT2 gives 81%, so the claim is understated and inconsistent.
- **§3.2, after Eq. 6:** −η⟨ḡ_j, ḡ_k⟩ > 0 needs **negatively** aligned gradients; the text says "positively".
