# Design notes

What the README summarises, with the reasoning and the evidence. Numbers come from three
kinds of sources, in this order of trust:

1. `results/**/metrics_*.json`, written by `train.py`, with git commit, `git_dirty` and the md5
   of every input file;
2. saved outputs of the versioned notebooks (single Colab runs, no seed record beyond what
   the code sets);
3. the LaTeX report (`rapport/`), produced from earlier runs whose logs were not kept.
   The report is left as delivered.

When two sources disagree, the README quotes the more trusted one and says so.

## 1. Model

**Structure.** Three CTFusion stages run a CNN sub-module and a Transformer sub-module in
parallel on the same input, concatenate them and max-pool over time. Each stage halves the
time axis and doubles the channels: (B, 10, 36) → (20, 18) → (40, 9) → (80, 4). Then come a
global max pool and a linear classifier `Linear(80 → n_classes)`.

**ALPE** (stage 1 only). `ALPE(t) = ECA(Conv1D(PE(t) ⊙ mask))`: the sinusoidal positional
encoding is zeroed on missing dates (cloud-masked acquisitions, `mask = 0`), passed through a
learnable 1-D convolution and an ECA channel attention, then combined with the input. A
missing date therefore carries no positional signal, instead of a position that pretends
the observation exists. Stages 2 and 3 work on pooled sequences and use no ALPE.
`tests/test_transformer.py` checks that the mask is required whenever ALPE is enabled.

**FFN width.** The paper gives 55,059 parameters for Arkansas but not the Transformer FFN
width. Two candidates were measured (`docs/doc_mctnet.md`, internal notes):

| FFN dim | Arkansas params | vs paper |
|---|---|---|
| 4×C | 39,718 | −15,341 |
| **8×C** (kept) | **56,798** | **+1,739 (+3%)** |

California has 56,879 parameters: one more class in the output layer, so 80 weights + 1 bias
= 81 more.

**GatedMCTNet** replaces the concatenation with a learned gate between the two branches:
+4,270 parameters (61,068), checked by `tests/test_gated_mctnet.py`. It lives in the same
`src/` and trains with `train.py --model gated`.

## 2. Training protocol

The paper's Table 3 fixes Adam, lr 0.001, n_head 5, kernel 3, 3 stages. On top of it,
`train.py` uses batch 32 and up to 200 epochs, plus three things the paper does not mention.
All three were added in `c2a8fcc` (2026-05-09), before the report was delivered on 2026-05-20:

- weight decay 1e-4;
- `ReduceLROnPlateau(factor=0.5, patience=10)` on the validation loss;
- early stopping on validation macro-F1, patience 20; the best checkpoint is evaluated on test.

The selection metric is macro-F1 because the California test set is unbalanced (rice 41%,
pistachios 6% on `scale30`). The Arkansas `scale30` test set is balanced (1,700 per class).

**Determinism.** `--seed` seeds `random`, NumPy and torch. On CPU, a rerun with the same seed
gave the same metrics to the last digit (Arkansas seed 42: 0.9776 twice). The thesis runs
were on a Colab GPU, which is not bitwise deterministic. This is one reason a same-seed CPU
run on `partie1` (0.9586) does not give the thesis's 0.9603 exactly.

**`git_dirty`** is computed with `git status --porcelain -- . ':!results'` inside the Point 5
folder. Changes to tracked code and untracked code files mark the run dirty; the run's own
outputs in `results/` do not. A first version also counted `results/` and flagged 5 of 6 runs
as dirty; those files were deleted and the runs repeated (`ec7cf7e`).

## 3. Data variants

All variants come from the same GEE export of 2021 (Sentinel-2 SR harmonized, 10 bands, 36
dates; labels from USDA CDL 2021, cropland confirmed by ESA WorldCover; our own zones, five
in Arkansas and eight in California, listed in `Point 2/point2.md`). Training uses 240
pixels per class. The variants differ in preprocessing and in the split:

| variant | test size AR / CA | used by |
|---|---|---|
| `scale30` | 8,500 / 8,200 | `train.py` default, README headline |
| `partie1` | 8,523 / 8,226 | thesis baseline (inferred, §4), GatedMCTNet, USkip, Multiscale |
| `scale20` | 8,500 / 8,200 | 20 m comparison, no versioned run |
| Part II pipeline (`TES_preprocessesd`) | 8,500 / 8,200 | covariate ablation, UNetMCTNetWithCovars |

`partie1` and `scale30` are different files for both regions (different md5, recorded in
`provenance.data_md5`), with different test sizes and class balance. How `partie1` was
produced is not documented in the repository beyond its name. The Part II pipeline reads different CSVs
(`arkansas_final.csv`, `California_10k_targeted.csv` with covariate columns), splits first,
then fits the covariate scaler on train only.

## 4. Reconciliation with the report

**Paper targets.** The report (`partie1.tex` l.406–410, `chapitre3_modele.tex` l.190–191)
attributes 0.9598 / 0.9421 (Arkansas) and 0.9502 / 0.9313 (California) to Wang et al. These
values do not appear in the paper. Its Table 5 (p. 7) gives OA / Kappa / F1 = 0.968 / 0.951 /
0.933 and 0.852 / 0.806 / 0.829. The internal notes (`resume.md`, `point1.md`, `point5.md`)
had the correct values. Consequence: the report's "our OA slightly exceeds the paper" for
Arkansas is not supported. The thesis run is 0.8 points below the paper, and the comparison is
not like-for-like anyway.

**Baseline.** Three seeds on `partie1` with the current `train.py`:

| | OA | Kappa | F1 | thesis |
|---|---|---|---|---|
| Arkansas | 0.9626 ± 0.0037 [0.9586, 0.9660] | 0.9532 ± 0.0047 | 0.9628 ± 0.0037 | 0.9603 / 0.9504 / 0.9603 |
| California | 0.9331 ± 0.0032 [0.9300, 0.9364] | 0.9191 ± 0.0039 | 0.9354 ± 0.0035 | 0.9311 / 0.9166 / 0.9339 |

All six thesis values fall inside the seed range. The training code did not change between
2026-05-09 and delivery. `GatedMCTNet.ipynb`, which trains "MCTNet vs GatedMCTNet", reads
`.../partie1` with the same configuration and seed 42. Hence the conclusion that the thesis
baseline is reproduced on `partie1`.

**Part II table.** `partie2.tex` l.239–257 differs from the versioned notebook
`MCTNetWithCovars.ipynb` on all 14 values. The best Arkansas configuration is `soil` (0.9775)
in the report and `clim_topo` (0.9799) in the notebook. The README quotes the notebook.

**USkip parameters.** The notebook (cell 6) builds its own MCTNet with a 4×C FFN (39,718)
and reports USkip at 50,842 (+11,124). The report's "−5,956 vs MCTNet" subtracts it from
the 8×C model (56,798), which is a different baseline.

**Multiscale.** The notebook holds two runs on `../data_train` (test sizes match `partie1`):
200 epochs without early stopping (0.9668 / 0.9322, cited by the report) and with early
stopping (0.9640 / 0.9233). Only the second follows the baseline protocol. The scripts it
calls, `train_visu.py` and `train_early.py`, are not in the repository; `multiscale/src/`
holds a `train.py` with a different interface.

**GatedMCTNet.** The notebook has no saved outputs. Its numbers (0.9695 / 0.9237) exist only
in the report and in `rapport/figures/courbes_gated.png`.

## 5. Limits and open questions

**Covariates against `scale30`: an open question.** The Part II runs have the same test sizes
as `scale30` (8,500 / 8,200). If their split were the same, the comparison would be:

- Arkansas: best covariate configuration 0.9799 against 0.9756 ± 0.0038 without covariates,
  about one standard deviation;
- California: every covariate configuration (0.9302 to 0.9423) below 0.9612.

That would contradict the report's "+1.72 points from soil". **This is not established.**
The Part II `.npy` files are not in the repository, so the split cannot be compared. The
California CSV is a different file (`California_10k_targeted.csv`), which makes an identical
split unlikely. The README therefore states neither a gain nor a loss. The test that would
settle it: run `train.py` (no covariates) on the Part II `.npy`, three seeds.

**Other limits.**

- Parts II and III are single runs. The baseline's seed std (≈ 0.003–0.004 OA) is the scale
  against which their deltas should be read.
- Small balanced training samples, one year, our zones: in-distribution numbers, not a
  mapping accuracy (the paper itself shows that mapping accuracy can differ from test
  accuracy).
- Notebooks hard-code the team's Drive paths and require Colab.
- Tests cover the model only (21 cases, offline, a few seconds, CI on Ubuntu and Windows).
  Parameter counts are pinned: exact per stage for CTFusion and the Transformer, exact for
  the GatedMCTNet overhead (+4,270), ±1% around 56,798 / 56,879 for MCTNet. A 4×C FFN fails
  three of them. Until 2026-10-03, the MCTNet check was ±10,000 around the paper's 55,059,
  and the CTFusion and Transformer checks asserted nothing. Nothing tests the data pipeline
  or `train.py`.

## 6. Reproducing the figures

- `python docs/make_figures.py` rebuilds `docs/results.png` and `docs/confusion_scale30.png`
  from the metrics files. The only typed-in numbers are the thesis and paper rows, each with
  its source in a comment.
- `python scripts/export_diagram.py` renders the README's Mermaid block to
  `docs/architecture.svg`. It needs Playwright with Chromium, which is not a project
  dependency.
