# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Two things live here:

1. **`Project_Baseball_Pitching/` — the active project.** Research + experiment harness for the
   LG Aimers 9기 × LG Twins hackathon: predicting a pitcher's **next-pitch command-success
   probability** (pitch-level binary probability, train 2019–2024 → hidden test 2025, anonymized
   IDs, metric assumed LogLoss). Task spec: `Project_Baseball_Pitching/HACKATHON_TASK.md`.
2. **Lecture materials (`1_`–`6_` folders)** — Korean-language LG AI curriculum slides (PDFs) plus
   one runnable notebook. Reference material only; see the bottom section. **Not in the public
   snapshot** (copyrighted course material) — if the folders are absent, you are in the public copy.

**Public release (2026-09-21).** (The `publish/` tooling below lives only in the private original; this public snapshot omits it.) DACON confirmed the team is *not* a Phase 3 finalist and that
participant-written code may be published as long as competition-provided data/files and personal
or sensitive information are excluded. `publish/export.py` builds a cleaned single-commit snapshot of
each of the four org repos into `publish/_export/<name>-public/` (exclusions + redactions + gates),
`publish/verify_public.py` re-lists the pushed tree, and `publish/TEAM_NOTICE.md` is the note sent to
teammates before their repos are flipped public. The original private repos are never rewritten.

## The project (`Project_Baseball_Pitching/`)

### Documents (read before adding research)

- `HACKATHON_TASK.md` — task spec; open questions (label definition, metric) tracked here
- `01`–`04` — landscape survey, playbook, LLM+RL feasibility verdict, cross-domain methods
- `05_final_blueprint.md` — integrated design (GBDT anchor + new-signal branches)
- `06_candidate_pipelines_catalog.md` — **the experiment menu**: models M0–M36, features F1–F23,
  targets T-a–f, calibration C1–C6, ensembling E1–E4, protocol arms P1–P8; sweep order S0–S7
- `07`–`08` — LLM-as-teacher research, teacher→student distillation plan (D1–D4, students S1–S4)
- `09_new_candidates_and_related_research.md` — 5-way parallel research (new models, baseball
  command metrics, temporal-holdout competition techniques, calibration, cross-domain/cold-start)
- `papers/` — ~34 arXiv PDFs; paywalled papers listed in `papers/PAPERS.md`

### Runnable code — `sweep/` (modeling) and `submission/` (packaging + compliance)

```bash
# every submission must clear this first — leaderboard-inversion signature scan
python submission/scan_probe_provenance.py submission/dist/<zip>     # BLOCK must be 0

python sweep/test_regulation.py                                       # §5 row-independence + parity
python submission/leg_matrix.py --auto --rows 30000 --anchor cregime  # leg RMS matrix, 0 slots
python submission/build_team_blend.py --tag <t> --leg name=<zip> ...  # assemble a blend
python submission/verify_submission.py <zip> --proxy 245789 --threads 6
```

`sweep/` is a flat package (~75 modules, sibling imports — do not nest it):
`real_data.py` (loader + byte-copied SERVE feature block + folds) · `lgbm_family.py` (june853
recipe) · `inseason.py` (in-season decomposition, the single biggest signal) · `eb_carrier.py`
(ENS-9 assembly + `DEPLOY_CAL`) · `tm_physmix.py` · `trackman*.py` (entity matching) ·
`dsf_*.py` (added 2026-08-13) · `test_regulation.py`. The synthetic-era modules live in
`sweep/_retired/`.

`submission/` holds the packaging and compliance tooling. Three of these are new as of
2026-08-29 and are the ones to reach for first: `scan_probe_provenance.py`,
`leg_matrix.py` (its `patch_meta_deep()` finds season-end lookups at any nesting depth —
the shallow version silently skips teammate packages and inflates RMS), and
`build_team_blend.py` (bundles N submission zips as byte-identical legs, each run under the
server contract in its own subprocess, then probability-averaged by `row_id`).

Available libs: numpy/pandas/sklearn/scipy/torch **+ lightgbm/xgboost/catboost**. The eval
server preinstalls torch 2.7.1/sklearn 1.8.0 and allows internet **during package install**, so
any PyPI package is fair game; install failures do not consume a submission slot, but exceeding
the 10-minute *inference* limit does. Real data lives in `Project_Baseball_Pitching/data/`
(gitignored); teammate repos are cloned into `partners/` (gitignored — clone with
`-c core.autocrlf=false` or newline conversion silently corrupts LightGBM text boosters).

### Non-negotiable principles (baked into docs and harness — do not "optimize" them away)

- **Evidence = prior, not verdict. No hard exclusion of any candidate.** Tiers (T1–T5) only
  allocate budget (~70–80% high-prior, ~20% long-shots run under fair conditions).
- **Never bake a leaderboard-inverted constant into a submission.** Public scores may be used
  to *select* among pre-registered candidates; solving for an optimum and hardcoding it is a
  §5 violation ("평가 데이터 전체를 보고 만든 사후 보정값"). This rule was broken in August
  2026 and cost the team its top ~1075 line — see `phase3/COMPLIANCE_DISCLOSURE.md`.
- **Adoption is decided by the time-split CV gate** (Brier Skill Score on real data; the older
  LogLoss framing predates the appendix), never by paper claims. A new feature arm ships only
  if `3-fold mean > 0` **and** `mean > across-fold SD` — random 6-column placebos clear a
  single-fold 2SE bar.
- **Zero leakage**: as-of features are strictly past-only (shift(1)); forward-chaining by season
  + game-level grouping; no random KFold; no resampling/SMOTE (calibrate instead); the target
  pitch's own Trackman measurements must never be inputs.
- **Teacher ≠ submission**: big models (LLM/TFM) are training tools only; the deliverable is a
  self-trained student that must still beat GBDT on the gate.
- Citations: prefer 2026→2025 papers, verify arXiv IDs actually exist (past research turned up
  reversed citations and vendor blogs disguised as papers).

### Current state / next steps (2026-08-30)

**▶ THE SCORE PURSUIT IS OVER (08-30 DACON ground-truth check — `docs/log/37_resume_runbook.md`
§6).** The Phase-3 cut is the verified top ~100 of the PRIVATE board, Private = the Public
score at competition end (officially no split shift), and **rank 100 currently sits at
1,135.19 and rising** — our team (ABS깡통존) is at rank 269 with 1075.84 (which is the
BLOCKED lineage; compliant best is 1058.60). Every card is arithmetically dead: self-built
combos (62,985 enumerated, 0 ≥ even 1100), and the six teammate zips (best case ~1100-1105,
undownloadable, memos empty) all fall far below ~1135+. Submission deadline 09-01; remaining
slots stay unused by default (no positive-arithmetic candidate exists). Certification
(Public ≥ 549.51) is long secured. DACON actively monitors submitted code and disqualifies
§5 violators (08-13 notice); our blocked line sits outside the mandatory-verification top-100,
and the compliance record (`phase3/COMPLIANCE_DISCLOSURE.md`) is in order.
Team best = **`submit_blendD4.zip` 1058.6047851923** (clookup+cmoe+physmix+tm3L, equal 1/4;
back-calc S(tm3L)=940.31). The blocked all-time high is 1075.8374148399 (2026-08-28 re-audit:
leaderboard-inverted constants in the champion lineage — the whole 1035–1075 line inherits
them, and it stays excluded regardless).

Why closed — the last unmeasured d-source (training-window/era betting) was measured twice and
died: the fit≤2023 gate said d(D4)=0.0437 ("wall broken"), but that yardstick was wrong.
**Measurement-convention law: d/gain feed `expected = mean(leg LB) + proxy_gain/1.4295` ONLY
when measured deployment-style** (models trained through 2024, leg_matrix p23-patched lookups)
— the ρ=1.4295 anchor (A3: 42.423→29.677) is in that convention; comparing a differently-
windowed model against the caches drops the shared in-sample component and inflates d ~1.5×.
Official deployment-convention measurement of the all-season leg (`submit_era_all.zip`,
2019-24 uniform LGBM×5 on XA78): **d(D4)=0.0286, E₅=1035.2 (−23.4 vs D4)** — breakeven
S=875 vs transfer 758. Fifth confirmation of the honest-d wall; the closure statement is the
breakeven curve `S_be(d) = 5·(1058.60 − (59.42+64000·d²)/1.4295) − 4026.56` with no candidate
above it (1100 still needs S≥1000 & d≥0.0344). legP (T1.5, local 825.5) is preserved but
shelved — same yardstick flaw, deployed d would be smaller. `submit_era_all.zip` stays as a
scan-CLEAN, verify-passed pre-registered asset (arithmetic-negative, do not submit).

08-30 addendum: the external-pretrained (HF) channel is also measured-closed
(`docs/research/16_hf_external_survey.md`). TabPFN 2.5/2.6/3 are license/token-blocked;
TabICLv2 — the only clean candidate (BSD-3, synthetic-only) — was piloted
(`sweep/tabicl_pilot.py`): paired accuracy vs same-context LGBM survived (+133±226, the
first FM to pass that gate here) but it is transductive (solo-vs-batch predictions differ
by 4.8e-2, deterministically = §5 test-row mixing) and ~8× over the 10-min limit — both
architecture-level, unfixable. The row-independence triple probe (solo/batch/repeat) in
that pilot is the standard first check for any future model candidate. A final full
re-enumeration (combo_search, 4,082 combos) then found exactly one candidate above the
champion — D4+cregime equal 1/5, expected 1059.20 (+0.60) — built as `submit_blendF5.zip`
(scan BLOCK 0, verify passed 264.1s) and submitted 08-30: **measured 1056.7903849842**,
−2.41 vs prediction (within error bars: realized variance gain 94.6% of predicted, or
S(tm3L)→928.3 within its ±10). Champion unchanged at blendD4 1058.60; with the last
positive-expectation candidate measured negative, the score pursuit is now conclusively
closed. (Do not confuse this blendF5 with the never-built era "blendF5 1035" in older notes.)

Remaining work on score: none. If any future measurement is ever needed, the loop is: build
deployment-style zip → `scan_probe_provenance` → `leg_matrix` (p23 convention) →
`combo_search` → E-formula. **The live work is now Phase 3 — see the next section.**

## Phase 3 (2026-09-03 ~ 09-07) — superseded

**▶ 2026-09-21: DACON confirmed ABS깡통존 was not among the Phase 3 teams.** Nothing below was
submitted; the package, PPT and email draft stay as a record of the reproduction work. The section
is kept verbatim for context. The live work is now the public release (see the top of this file).

The score chase is over; **Phase 3 code + PPT submission is the live work.** The user confirmed
advancement. Deliverable rules (competition 236743 `overview/rules` §3, transcribed verbatim in
`phase3/RULES_PHASE3.md`): four required items — training-environment/library versions,
**Private Score 재현용 학습 코드** (inference code is replaced by the leaderboard submission),
a free-form solution PPT, and each member's offline-hackathon attendance — emailed to
`dacon@dacon.io`, code as `.py`/`.ipynb` in UTF-8, **due 09-07 10:00**, verification through 09-11.

Verification target = `submit_blendD4.zip` (1058.6047851923). Build the package with
`python phase3/tools/collect_package.py` (208 files, 11.6MB zip; `phase3/package/` and
`phase3/dist/` are gitignored). The collector mirrors repo-relative paths on purpose — `sweep/` is
a flat sibling-import package and resolves `data/` and `results/` via `parent.parent`.

**Reproduction is measured, not asserted** (09-03, run inside the package tree): physmix retrain
reproduces all four model weight files bit-identically and `pm_corrector` exactly; tm3L reproduces
all 5 entries bit-identically; the reassembled blendD4 matches the original on 245,789 rows with
**RMS 0.000000**. The three remaining zip-entry differences are the serving template gaining
inactive ENS-10/11 branches, one provenance sentence corrected during the compliance fix, and the
MANIFEST that records both.

Two hazards found while doing this, both now guarded in `collect_package.py`:
- **Shared-artifact drift.** The 08-30 TrackMan redesign (GMM tier re-cut, commit `3917b5f`)
  overwrote `results/trackman/matches.csv` in place, dropping Tier-1 from 407 to 179 pitchers, which
  silently broke physmix reproduction. The submission used commit `17fc655`; the collector pins that
  version. Treat shared result artifacts as versioned inputs, not scratch space.
- **`git show` is not a safe way to copy binaries.** It returns a 132-byte LFS pointer for
  LFS-tracked files (JTT's `submit_v3.zip`) and can normalize newlines. The collector asserts each
  partner worktree is clean and at its pinned commit, then copies bytes from disk.
- Also: `shutil.rmtree` over a tree containing a `data/data` junction can delete the real dataset.
  `safe_rmtree()` unlinks reparse points first.

Remaining before the deadline: teammate answers (`phase3/TEAM_REQUEST.md` — attendance, real names,
redistribution consent, their own reproduction report), the PPT (`phase3/PPT_OUTLINE.md` → .pptx),
and the submission email (`phase3/EMAIL_DRAFT.md`). The board's team-representative score may be the
banned 1075.84 lineage; we disclose that up front in the email and in
`phase3/COMPLIANCE_DISCLOSURE.md` rather than defending it.

Read `docs/log/33` + `35` for the incident and recovery arithmetic; ledger =
`results/lb_history_260829.md`. Build/measure loop for any future leg: build deployment-style
zip → `scan_probe_provenance` (BLOCK 0) → `leg_matrix` (p23 convention) → E-formula →
`build_team_blend` → verify → submit one. The serving template now supports lgbm
`input:"full"` members (78-col XA legs — `submission/build_era_leg.py` is the worked example).

## Environment notes

- Windows 11 + PowerShell. Filenames contain Korean, spaces, and full-width brackets (`『』`) —
  always quote paths. Read files with explicit `encoding='utf-8'` (cp949 default will break).
- GitHub: `origin` = `LGAimers-9th-Hufs/LG-Aimers-Pitch-Command-Hackathon` (team org, private);
  `backup` = `oks706/...` (personal, pushed separately). Three teammate repos live in the same
  org and are cloned into `partners/`. Keep `.zip` archives out of commits.
- Project docs and commit messages are written in Korean.

## Lecture materials (reference)

Six topic folders (`1_Mathematics_for_ML` … `6_Optimization_DFL_TimeSeries`), indexed in
`INDEX.md` with per-folder `_RESEARCH.md` reports and a root `RESEARCH_SUMMARY.md`. The one
executable teaching artifact is `3_Tabular_ML/00_Hands_on_Tabular_ML.ipynb` (Colab T4): one
binary-classification dataset run through LogReg → XGBoost → TabM → LLM serialization → TabPFN;
cells run top-to-bottom and later cells depend on earlier variables. Its flow mirrors the
project's model families and is the quickest hands-on reference.
