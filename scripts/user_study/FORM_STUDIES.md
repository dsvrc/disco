# Google Forms user studies (TMR paper, supplementary H.12 and H.15)

Two studies, one Google Form each, every participant signed in with a Google account (verified e-mail = record of who
took part). Each participant sees 12 videos in Study 1 and 15 in Study 2. Each study has two groups (versions) with
different stimuli, chosen by the same rule. Participants get Group 1 or Group 2 in alternation, so twice as many
stimuli are rated without making the forms longer. How to build the forms: `GOOGLE_FORMS_HOWTO.md`.

| | Study 1: DiscoForcing preference (H.15) | Study 2: StyleRetention validation (H.12) |
|---|---|---|
| Question | Do people prefer TMR+KD's dance to fine-tuning's, and does it keep the earlier genres? | Do people's style judgments follow StyleRetention, and more closely than the existing metrics? |
| Method | blind paired comparison (2AFC), three criteria | reference-based rating (5-point), one test clip at a time |
| Videos per form | 8 retention + 2 newest-genre trials + 2 attention checks = 12 | 12 test clips + 3 controls = 15 |
| Video | 15 s with music: real dance (reference), A, B | 5 s silent clip shown twice: 2 real reference clips, TEST |
| Time | about 10 min | about 7 min |
| Stimuli from | `disco_study_pack.zip` (server, 2026-11-18) | the existing `study_pack` |
| Rendered | `form_studies\study1_disco` (24 videos) | `form_studies\study2_style` (30 videos) |

Everything random is drawn from fixed seeds by the scripts, before anyone has looked at any output. No clip is
picked or swapped by hand. The only filter is the automatic sanity check: no NaN, no joint more than 1.5 m from the
pelvis, root slower than 6 m/s.

## Study 1: TMR+KD against fine-tuning on DiscoForcing

**Hypotheses.** H1 (primary): on test music of earlier genres after the final session, people judge TMR+KD's dance
closer in style to the real dance more often than fine-tuning's (preference > 50%). H2 (secondary): the same on music
and naturalness. H3 (secondary, non-inferiority): on the newest session's genres (Ballet Jazz and Street Jazz, after
Session 4), TMR+KD's preference is no worse than 40% (margin 10 points), i.e. replay does not cost the newest genres.

**Trials per form.** Three kinds:
- 8 *retention* trials per group: after Session 4, one per earlier genre (Lock, Waack, House, Middle Hip-hop, LA
  Hip-hop, Pop, Break, Krump), on that genre's test music. Each group has its own draw.
- 2 *newest-genre* trials: Ballet Jazz and Street Jazz after Session 4. Both groups see both.
- 2 *attention checks*: one option is the reference dance itself, the other a real dance of a genre two or more
  sessions away. Both groups see both.

AIST++ has a single test music piece per genre, and its short "basic" sequences (7 to 12 s) cannot hold a 15 s
window. A form can therefore hold only one trial per genre. The pack (`pack_disco_study.py`) skips the planned
plasticity trials of Sessions 1 to 3 (their songs are taken by the retention trials) and finds a song for only one
check per group. `render_form_studies.py` shares the newest-genre trials and the checks between the groups; this is
the H.15 plan of rating the newest session's music. It also mirrors one check so that the correct answers alternate
(B, A), and orders each form so that no two neighbouring trials play the same song.

For each trial, the pack drew a sequence of the genre's test song, a generation sample and a 15 s window starting at
least 3 s in. Both models use the same sample, so the same noise seed. The reference is the real dance of that
sequence, synchronized with the same music. TMR is on side A in 4 of the 8 retention trials of each form, and in one
of the two newest-genre trials. Checks are at positions 5 and 10. GCR+KD is packed on the same windows, so a TMR+KD vs
GCR+KD form can be added later without a server trip.

**Questions per trial** (forced choice, required):
1. Style: which dance, A or B, is closer to the STYLE of the real dance on the left?
2. Music: which dance, A or B, moves better WITH THE MUSIC (rhythm, beats, energy)?
3. Naturalness: which dance, A or B, looks more NATURAL as human movement?

**Exclusion** (fixed now): a participant who gets the style question of either attention check wrong, or leaves the
form incomplete. A participant who reports no sound is left out of the music question only.

**Analysis** (fixed now):
- Primary: the share of retention-trial style choices that favor TMR+KD, with a 95% CI from a two-way bootstrap that
  resamples participants and trials (2,000 resamples). H1 holds if the CI lies above 50%.
- Also reported: the naive binomial test over all choices (for comparison with other papers), the same share for
  music and naturalness, and the newest-genre trials (H3: lower bound above 40%).
- Agreement between raters: Fleiss' kappa per criterion.
- Exploratory: the correlation over trials between the human style preference and the difference in StyleRetention
  between the two shown clips (in `answer_key.csv`). This tests SR at the level of models.

## Study 2: does StyleRetention agree with people?

**Hypotheses.** H1 (primary): the mean human style rating of a clip (MOS) increases with its StyleRetention
P(music's genre) (Spearman rho > 0). H2: rho is higher for StyleRetention than for the handcrafted genre classifier and
the per-clip FID_k / FID_g analogues. H3: 1 - P(other) separates clips people call dancing from those they do not
(AUC > 0.5).

**Clips** (`render_form_studies.py style`, seed 0): each version has 12 generated 5 s windows, taken from the 396
windows of the existing pack (all methods and checkpoints). The music is distinct, with at most 2 clips per genre and
3 per method. Of all such sets, the script keeps the one whose scores best cover each metric's range and are least
correlated with one another. The five scores are SR, the handcrafted classifier and 1 - P(other) (each spread over
[0, 1]), and the kinetic and geometric distances (spread over their percentiles). The rule treats all metrics alike,
and it makes them disagree on many clips, so the ratings can show which metric people follow (maximum
differentiation, Wang & Simoncelli 2008). Rendered result: SR spans 0.00 to 0.99 in both versions, and its rank
correlation with the other scores within a version is between -0.09 and +0.31.

Each version also has three controls, the same in both versions:
- a hidden reference: another real clip of the reference genre (expected rating high, dancing);
- a real clip of a genre from a distant session (expected rating low);
- a figure standing still (expected not dancing).

**Questions per clip** (required): (1) "How close is the dance STYLE of the TEST figure to the two REFERENCE
dances?" on a scale from 1 (completely different style, or not dancing) to 5 (same style). (2) "Is the TEST figure
dancing?" Yes / No / Not sure.

**Exclusion** (fixed now): exclude a participant who rates the distant-genre control or the static control at least as
high as the hidden reference, who answers Yes for the static figure, or who leaves the form incomplete.

**Analysis** (fixed now): MOS = mean rating of the included raters of a clip's version. Over the 24 test clips:
- Primary: Spearman rho(SR genre level, MOS), with a 95% CI from a two-way bootstrap over clips and raters.
- Also reported: Kendall tau-b, and the share of clip pairs ordered as people order them.
- Secondary: the same for the handcrafted classifier, the kinetic and geometric distances, style-embedding cosine,
  1 - P(other) and session-level SR, each with a paired bootstrap of rho_SR - rho_metric.
- Ceiling: split-half reliability of MOS (Spearman-Brown corrected).
- Agreement between raters: Krippendorff's alpha (ordinal).
- H3: the AUC of 1 - P(other) against the majority answer to "dancing".
- Robustness: repeat with ratings z-scored per rater.

Controls are never in the metric analysis.

## Participants, consent and the forms

- Target 24 to 30 adults per study (12 to 15 per version). The same people can take both studies. Exclude anyone
  who knows the methods (co-authors, people who have seen the results).
- First page of each form: purpose, time, voluntary, can stop any time, no compensation (or state it), and that
  **Google Forms records the e-mail address of the Google account to allow one response per person and to document
  participation; it is stored separately from the ratings, never published, and deleted after the review period**.
  Then the consent checkbox (18 or older, agree). Then two profile questions: dance experience (none / casual /
  trained) and whether they can tell street styles apart. Study 1 adds the device and whether sound is on.
- Settings: Collect e-mail addresses = Verified, Limit to 1 response, all questions required, no shuffling (the
  order is in the videos), progress bar on, results not shown to respondents.
- Videos: upload to YouTube as **Unlisted** with neutral titles (the file names), then embed them in the form.
  Google Forms can only embed YouTube videos. Use a channel without your name if possible, since participants see
  it in the embed.
- Upload only the `.mp4` files. Never upload `answer_key.csv`, `study.json` or `pack_log.txt`: they say which model
  is A or B. The `.sig` files next to the videos let a rerun re-render only videos whose design changed.

## What the paper needs once the forms are fixed

- H.12 currently describes the older 180-pair site study (anonymous, no e-mail). Its design paragraph and Tab. 26
  must describe this rating study: rho with 95% CI per metric, plus the ceiling.
- H.15 / Tab. 28 currently plan a BEAT2 + SAM rating study of four methods. They become DiscoForcing TMR+KD vs
  fine-tuning preference: rows = criterion x condition, columns = preference for TMR+KD, 95% CI, n.
- **Sec. M (ethics) must change.** It says the studies collected no personal or identifying information and that
  responses cannot be linked to anyone. Collecting e-mails makes both false. Replace that with the e-mail sentence
  above, and check whether your institute requires ethics approval when e-mails are collected.

## Commands

Server (repo root, any GPU-free moment; CPU only, a few minutes):

    python scripts/user_study/pack_disco_study.py --results results/aistpp_genre5/screen/v2_40000_full_s0v2

It writes `disco_study_pack.zip` (tens of MB; the music is cut to the windows) and prints every trial. Missing dances
are listed with the exact `eval_cl_session.py` command that regenerates them.

Laptop (renders into `C:\Users\chinnu\3d_cvpr\form_studies`; both done 2026-11-18/19), then the Apps Script:

    python scripts/user_study/render_form_studies.py disco --pack C:/Users/chinnu/3d_cvpr/disco_study_pack.zip --out C:/Users/chinnu/3d_cvpr/form_studies --workers 2
    python scripts/user_study/render_form_studies.py style --pack C:/Users/chinnu/3d_cvpr/study_pack --out C:/Users/chinnu/3d_cvpr/form_studies
    python scripts/user_study/make_forms_script.py --studies C:/Users/chinnu/3d_cvpr/form_studies --out scripts/user_study/google_forms_builder.gs

Style encoder accuracy on real AIST++ dance (server, repo root; CPU). It writes
`raw_data/cl_sessions/_style_encoder/aistpp/validation.json` and fills the encoder column of Tab. sr_valid:

    STYLE_ENCODER_DEVICE=cpu python -m metrics.style_encoder validate --dataset aistpp 2>&1 | tee encoder_validate.log

`validate` scores sessions with the benchmark's current pairing, and says so if the checkpoint or `eval/probe.npz`
stored an older one. It reports window-level genre and session accuracy on the test music, which no training dance
used, per-genre accuracy and confusions, whole dances assigned by majority vote, accuracy on dancers never seen in
training (3 folds; it trains a missing fold encoder first), robustness to artefacts, the genre-swap retention
matrix, and rejection of non-dance.
