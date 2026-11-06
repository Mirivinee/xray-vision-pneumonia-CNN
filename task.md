# TASK.md — X-Ray Vision (Group 2): everything left to do for an excellent submission

> **Project:** Pneumonia detection from pediatric chest X-rays (Atria University, NN & CNN Sprint capstone)
> **Team:** Suraj · Anurag (leader) · Vineet · Meghana S
> **Written:** 2026-11-06, from a review of the brief (PDF), `gradcam_analysis.zip`, and the transfer-learning results shared in chat.
> **Goal:** meet every minimum requirement, score well on every "major focus" pointer in the brief, and add a few things that make the project stand out.

---

## 0. How to use this file

**Priority legend**

| Tag | Meaning |
|---|---|
| **P0** | Blocks a minimum requirement or invalidates results. Do first. |
| **P1** | Needed for strong marks (the brief's "major focus areas"). |
| **P2** | Depth and polish that separates good from excellent. |
| **P3** | Beyond expectations. Only after P0–P2 are done. |

**Ground rules (apply to every task)**

1. **The test set is frozen.** It has already been evaluated 5 times (3 scratch baselines + 2 transfer models). No more model, threshold, or hyper-parameter decisions from test results. The report must state this honestly.
2. **All selection on validation only** (model choice, imbalance strategy, threshold, epochs).
3. **One split, one seed policy, one preprocessing pipeline** shared by every model, so comparisons are like-for-like.
4. **Every number in the README/report comes from a saved file in `results/`** (CSV/JSON/PNG), never retyped from memory.
5. **Everyone commits from their own GitHub account**, on feature branches, spread across the timeline. No bulk upload.
6. Fill in the `Owner` of each task. The final contribution table must be truthful.

**Proposed ownership (edit freely — brief suggests two areas per member)**

| Area | Workstream | Proposed owner |
|---|---|---|
| Data, EDA, re-split | WS-A | ______ |
| Scratch CNN + imbalance | WS-B | ______ |
| Transfer learning | WS-C | ______ |
| Evaluation + threshold | WS-D | ______ |
| Grad-CAM + error analysis | WS-E | ______ |
| Repo, reproducibility, README | WS-F | ______ |
| Report (every member writes the part they owned) | WS-G | all |

---

## 1. Where things stand (snapshot)

### 1.1 Data (after exact-duplicate removal; official 5,856 → 5,824 images)

| Split | NORMAL | PNEUMONIA | Total | % pneumonia |
|---|---:|---:|---:|---:|
| train | 1,213 | 3,475 | 4,688 | 74.1% |
| val | 135 | 383 | 518 | 73.9% |
| test | 231 | 387 | 618 | 62.6% |

Majority-class accuracy is 73.9% on val and **62.6% on test**. Any model at or below that is useless.

### 1.2 Results so far (test set, 0.5 argmax unless noted)

| Model | Acc | Sensitivity | Specificity | Missed (FN) | False alarms (FP) | Notes |
|---|---:|---:|---:|---:|---:|---|
| Scratch CNN, class-weighted | 0.7735 | 0.9121 | 0.5411 | 34 | 106 | |
| Scratch CNN, oversampling | 0.6262 | 1.0000 | 0.0000 | 0 | 231 | **Broken run — see ISS-01** |
| Scratch CNN, augmentation only | 0.7201 | 0.9767 | 0.2900 | 9 | 164 | |
| ResNet50 (fine-tuned) | 0.8398 | 0.9897 | 0.5887 | 4 | 95 | threshold unclear (ISS-04) |
| DenseNet121 (fine-tuned) | 0.8689 | 0.9664 | 0.7056 | 13 | 68 | confusion matrix reconstructed from recall/precision; threshold = validation-F1 pick, value not recorded |

### 1.3 Coverage of the brief

| Brief requirement | Status | Tasks |
|---|---|---|
| EDA: class counts, samples, sizes, **per-class** intensity histograms | PARTIAL (histograms pooled) | A4 |
| Stratified re-split, justified; test untouched | PARTIAL (not patient-grouped; test lost 6 duplicates, undocumented) | A1–A3, A5 |
| Scratch CNN + learning curves | DONE (thin) | B2, B7 |
| Val metrics: acc, precision, recall, specificity, F1, ROC-AUC | **TODO** (test only, no AUC) | B6, D2 |
| ≥2 imbalance strategies compared + recommendation | PARTIAL (oversampling buggy; picked on test) | B1, B3, B4, B8 |
| Augmentation + why vertical flips are bad | PARTIAL (no brightness/contrast, no discussion) | B5 |
| ≥2 pretrained models, 1→3 channel handled | PARTIAL (trained; not compared properly on val) | C1–C3 |
| Frozen vs fine-tuned comparison | **TODO** | C3 |
| ROC + PR curves | **TODO** | D3 |
| Threshold chosen on val for high sensitivity, justified | **TODO** | D4 |
| Test confusion matrix + FN vs FP cost discussion | PARTIAL | D5, D6 |
| Grad-CAM on correct and incorrect images | **TODO** (never run; cannot run as written) | E1–E7 |
| README, contribution table, report, public repo from fresh clone | **TODO** | WS-F, WS-G |
| Visible contribution from all 4 members | UNKNOWN (check `git shortlog -sne`) | F10 |
| AI-assistant usage disclosed | **TODO** | F15, G |
| "Not a diagnostic tool" statement | **TODO** | F5, G |

---

## 2. Issues and problems log

Severity: **CRITICAL** = wrong or unusable results · **HIGH** = costs marks in a major focus area · **MEDIUM** = quality/rigor gap · **LOW** = polish.

### ISS-01 · CRITICAL · The oversampling "failure" is a pipeline bug
- **Problem:** the helper builds `class0.take(N).concatenate(class1.take(N))` and then `shuffle(buffer_size=1000)`. The concatenated stream is ordered by class and the buffer (1,000) is far smaller than the stream (6,950), so batches are almost single-class.
- **Evidence:** simulating this shuffle gave 79 of 218 batches all-NORMAL and 24 all-PNEUMONIA. The first ~30% of every epoch contains 0% pneumonia and the last ~40% is 72–98% pneumonia. This matches the logs: validation accuracy frozen at 0.7394 (= val pneumonia share), training accuracy *falling* 0.89 → 0.71, and the test model predicting PNEUMONIA for all 618 images.
- **Where the bug lives:** `imbalance.ipynb`, `Baseline_Cnn/baseline_cnn_oversampling.ipynb`, `docs/IMBALANCE_INTEGRATION.md`.
- **Consequence:** the README conclusion "oversampling did not work" is unsupported. Cause is likely the pipeline; confirm by re-running with the fix.
- **Fix:** B1.

### ISS-02 · CRITICAL · Grad-CAM cannot run as written (and has never run)
- All checkpoint paths are `None` and the notebook has zero outputs.
- `baseline_cnn_class_weighted.keras` (best scratch model) is **not in the zip**; its save cell was never executed.
- Loader asserts input shape `(None,224,224,1)` and uses `safe_mode=True`. Transfer models are 3-channel and contain a `Lambda` layer, which safe mode refuses to deserialize.
- `gradcam()` searches `model.layers` for a `Conv2D`; in transfer models the convs are nested inside the backbone sub-model, so the search fails.
- Example selection takes the *first* correct and *first* incorrect validation image. Files sort NORMAL-first, so both are NORMAL: no false negative is ever shown, and n = 1.
- Uses the softmax probability as the target score (saturates); logits are better.
- **Fix:** E1–E7.

### ISS-03 · HIGH · Model/strategy selection done on the test set
- The baseline README picks class weighting as "strongest" from test numbers. The transfer comparison (ResNet50 vs DenseNet121) is currently visible only on test. The brief says hyper-parameters and thresholds are chosen on validation only.
- On validation the class-weighted and augmentation baselines tie at 0.8822.
- **Fix:** B4, D2, D4. Disclose that test was evaluated 5 times before the freeze.

### ISS-04 · HIGH · Metrics and thresholds are not clinically meaningful yet
- Baselines are compared at a fixed 0.5 argmax. Sensitivity/specificity at one arbitrary cutoff says little about ranking quality.
- DenseNet121's threshold was picked by F1 over 5 coarse values (0.3–0.7); the brief asks for a threshold that **keeps sensitivity high**. ResNet50 got no threshold selection, and the two reports appear to use different evaluation code, so thresholds may differ.
- No ROC, PR curve, ROC-AUC, or PR-AUC anywhere in the project.
- Test prevalence (62.6%) differs from train/val (~74%), so PPV/NPV shift between validation and test.
- **Fix:** D1–D6.

### ISS-05 · HIGH · Probable patient-level leakage; validation likely optimistic
- Split is a random stratified split by *image*. The dataset contains several images per child (pneumonia filenames carry a person ID), so the same patient can sit in train and val.
- Symptoms: validation 0.88 vs test 0.77 for the baseline; transfer-model training accuracy 0.97–0.99 vs test 0.84–0.87. (Part of the gap may also be a genuine train/test distribution shift; the checks below separate the two.)
- Only a code comment in the Grad-CAM notebook mentions the risk; nothing is done about it.
- **Fix:** A2, A4 (adversarial validation), report limitations.

### ISS-06 · HIGH · Reporting gaps
- Baselines report test metrics only; nothing on validation (brief asks for validation metrics).
- Transfer-learning curves plot **training** accuracy/loss only; training loss is class-weighted and has augmentation + dropout active, so it is not comparable to test loss. No validation curves, no marker for where fine-tuning starts.
- README's "best logged validation accuracy" (class-weighted: 0.8822 at epoch 3) is not the tested model: `restore_best_weights` on val_loss restored epoch 8 (val acc 0.8687).
- **Fix:** B6, B7, C3, D2.

### ISS-07 · HIGH · Not enough experimental depth ("not one lucky run")
- Single seed (42), three baseline runs, no ablations, no hyper-parameter study.
- Baseline CNN is small (101k params), has no BatchNorm, ran at most 10 epochs (class-weighted run was still improving), and trained on **CPU** (70–140 s/epoch on native Windows TensorFlow). The brief points to free Colab/Kaggle GPUs.
- Test set has 618 images: a 95% interval on accuracy is about ±3 points, so many differences above are within noise. Each missed pneumonia case moves sensitivity by ~0.26 points.
- **Fix:** B2, B4, C4, H1.

### ISS-08 · HIGH · Transfer learning setup is not a fair, controlled comparison
- Transfer models use class weights **+** augmentation (incl. horizontal flip) **+** ImageNet preprocessing; each scratch baseline changed only one thing.
- Horizontal flip mirrors the heart to the right side: defensible (CheXNet used it) but must be justified or ablated. The brief only asks about vertical flips.
- Frozen-head vs fine-tuned: only the final model is kept; the head-only stage is never evaluated, so the required comparison cannot be made.
- No brightness/contrast augmentation (requested in the brief).
- `image_dataset_from_directory` squashes every image to 224×224 ignoring aspect ratio; originals range ~384–2,916 px wide, so shapes are distorted.
- 1→3 channel handling is implicit (default RGB loading replicates the gray channel). It must be explicit and documented.
- **Fix:** C1–C7.

### ISS-09 · MEDIUM · EDA is incomplete
- One pooled intensity histogram of 18 sampled images, not **per class** as required.
- Imbalance not quantified as ratios (train 2.9:1, test 1.7:1).
- Not checked: whether image size, aspect ratio, brightness, or grayscale-vs-RGB mode (283 images are RGB) correlates with the label. Any such correlation is a shortcut a CNN can learn.
- Text markers/labels (mentioned in the brief) are not examined.
- **Fix:** A4.

### ISS-10 · MEDIUM · Data preparation is undocumented and not fully reproducible
- Duplicate removal deleted ~32 files, including 6 from the "untouched" official test set (234/390 → 231/387). Defensible, but it must be documented with counts.
- The split notebook has no saved outputs, so counts/duplicates are not visible without re-running.
- Split lives only inside a notebook, with no committed manifest listing which file went where.
- Only exact (byte-identical) duplicates are caught; near-duplicates (re-encoded/resized) are not.
- **Fix:** A1, A3.

### ISS-11 · MEDIUM · Model files and checkpoints
- Class-weighted checkpoint missing (ISS-02).
- `ResNet50` has ~23.6M parameters; a full `.keras` save with optimizer state can approach or exceed GitHub's 100 MB file limit. Do not commit it.
- `Lambda`-based preprocessing makes saved models awkward to reload (ISS-02).
- **Fix:** C9, F11.

### ISS-12 · MEDIUM · Repository is far from the required structure
Missing: root `README.md`, `requirements.txt`, `.gitignore`, `src/`, `results/` (figures, `metrics.csv`), numbered notebooks, `report/`, contribution table. Only `Baseline_Cnn/README.md` exists.
**Fix:** WS-F.

### ISS-13 · MEDIUM · Stale and duplicated documentation/code
- `docs/IMBALANCE_INTEGRATION.md` says "there is currently no CNN implementation or training loop in this repository" (no longer true) and references an `imbalance.py` that does not exist.
- The helper functions are copy-pasted into several notebooks instead of living in `src/`, so a bug (ISS-01) exists in four places.
- **Fix:** F2, F12.

### ISS-14 · MEDIUM · Environment fragility
- Hard-coded Windows paths and a hard-coded KaggleHub cache path in the Grad-CAM notebook; each notebook has its own data-discovery logic.
- Library versions not pinned (saved models show Keras 3.12.4; TensorFlow version not recorded).
- **Fix:** F3, F7, F9.

### ISS-15 · MEDIUM · Contribution visibility and AI disclosure
- Every executed output in the zip comes from what looks like one Windows machine (`C:\Users\apsur`, `C:\dev\...`). The brief checks commit history for all four members and penalises a bulk upload.
- Several notebooks/docs read as AI-assisted (e.g. "the CNN teammate", "I found no saved model/checkpoint files"). That is allowed, but the report must say where and how, and every member must be able to explain every line.
- **Fix:** F10, F15, WS-G, WS-I.

### ISS-16 · LOW · Brief items nobody has addressed yet
Vertical-flip discussion · brightness/contrast augmentation · focal loss (optional third strategy) · explicit "not for clinical use" statement · related work (3–5 papers) · limitations/ethics/dataset-bias section.

### ISS-17 · LOW · Outputs not saved
Learning curves, split counts, imbalance runs, transfer runs, and Grad-CAM exist only inside notebooks (or nowhere). The brief wants outputs saved so they render on GitHub, plus figures in `results/`.
**Fix:** F6, F8.

---

## 3. Work plan

Format: `- [ ] **ID [Priority]** task — Owner: ____` followed by **Done when** criteria.

### WS-A · Data, EDA, split

- [ ] **A1 [P0]** Move split logic into `src/data.py` (deterministic, seeded) and write a **manifest** `data/splits/manifest.csv` with columns: `path, label, split, group_id, sha256, width, height, mode`. Commit the manifest (small), not the images. — Owner: ____
  - **Done when:** a fresh clone regenerates the identical manifest; all models read splits from it.
- [ ] **A2 [P0]** **Decide and implement patient-level splitting.** Derive group IDs from filenames (see Appendix A.2), use `StratifiedGroupKFold` for the train/val split, and audit group overlap between train/val **and** official train/test. — Owner: ____
  - **Done when:** zero group overlap between train and val; overlap with the official test set is counted and reported; number of images falling in the "unknown pattern" fallback is reported.
  - **Decision gate:** this changes every number. Decide first; if regrouping, re-run everything once on the final split. If there is truly no time, state it as a limitation in the report and keep the old split.
- [ ] **A3 [P1]** Document duplicate handling: counts before/after per split and class (5,856 → 5,824), how test was affected (6 images), and why copies are kept in train. Optional: near-duplicate check with perceptual hashing. — Owner: ____
  - **Done when:** a table in README/report plus `results/data_counts.csv`.
- [ ] **A4 [P1]** Complete the EDA (`notebooks/01_eda.ipynb`): — Owner: ____
  - [ ] class counts and **imbalance ratios** per split (CSV + figure)
  - [ ] **per-class** pixel-intensity histograms and mean image per class
  - [ ] width/height/aspect-ratio distributions **per class and per split**; test whether size or mode predicts the label (simple logistic regression on metadata)
  - [ ] grayscale vs RGB (283 RGB images) per class
  - [ ] brightness/contrast statistics train vs test
  - [ ] gallery of images with text markers/labels; estimate how common they are and in which class
  - [ ] **adversarial validation:** classifier to tell train from test images (AUC ≫ 0.5 means real distribution shift)
  - **Done when:** every finding is stated in one sentence and figures are saved in `results/figures/`.
- [ ] **A5 [P1]** Write the justification for the validation split (why 10%, stratified + grouped, why the official 16-image val folder is unusable). — Owner: ____
- [ ] **A6 [P2]** Preprocessing decisions in writing: 224 vs 256, aspect-ratio handling (squash vs `resize_with_pad`), normalization, optional CLAHE ablation. — Owner: ____

### WS-B · Scratch CNN and imbalance strategies

- [ ] **B1 [P0]** Fix oversampling with `sample_from_datasets` (Appendix A.1), put it in `src/`, delete the buggy copies, add a sanity check that batches hover near 50% positives, then re-run. — Owner: ____
  - **Done when:** the oversampling run no longer collapses to one class, the old result is explained in the report as a bug found and fixed, and the helper lives in exactly one place.
- [ ] **B2 [P1]** Strengthen and justify the baseline architecture. — Owner: ____
  - [ ] add BatchNorm; allow 30–50 epochs with early stopping on validation (patience 5–8) and a LR schedule
  - [ ] ablations: depth (3 vs 4 blocks), width, dropout (0.2/0.3/0.5), BN on/off, GAP vs Flatten
  - [ ] table justifying each design choice (3×3 filters, pooling, ReLU, GAP, dropout)
  - **Done when:** `results/ablation_baseline.csv` with val AUC, sensitivity@spec, parameters, epochs.
- [ ] **B3 [P1]** Add **focal loss** (γ ∈ {1, 2}) as a third imbalance strategy; also test **class weights + augmentation** as an explicitly documented combination. — Owner: ____
- [ ] **B4 [P0]** Compare strategies (class weights, fixed oversampling, focal, augmentation, best combination) with **≥3 seeds each**; report mean ± std on **validation**; choose by val ROC-AUC and sensitivity at a fixed specificity. — Owner: ____
  - **Done when:** one results table, no test numbers used for the choice.
- [ ] **B5 [P1]** Augmentation: add small brightness/contrast; keep rotation ≤ ~±7°; **write the discussion of why vertical flips are bad** (anatomy is not vertically symmetric: heart, stomach bubble, diaphragm; never occurs at test time) and discuss horizontal flip (dextrocardia argument vs CheXNet precedent), ideally with a with/without ablation. — Owner: ____
- [ ] **B6 [P0]** For every baseline, report on **validation**: accuracy, precision, recall (sensitivity), specificity, F1, ROC-AUC, PR-AUC. — Owner: ____
- [ ] **B7 [P1]** Save learning curves (train **and** val) as PNG for every run into `results/figures/`. — Owner: ____
- [ ] **B8 [P1]** Write the imbalance recommendation (a short paragraph with the evidence and the trade-off). — Owner: ____

### WS-C · Transfer learning

- [ ] **C1 [P0]** One shared recipe/config for scratch **and** transfer models (same split, class weighting, augmentation, seeds). Any difference must be an intentional, labelled ablation. — Owner: ____
- [ ] **C2 [P0]** Make the **1→3 channel** conversion explicit (load grayscale, then replicate with `tf.image.grayscale_to_rgb` or concatenation) and document it. — Owner: ____
- [ ] **C3 [P0]** **Frozen vs fine-tuned:** save a checkpoint (or predictions) after the head-only stage and evaluate it separately from the fine-tuned model, on validation, for both architectures. Plot train **and** val curves with a vertical line where fine-tuning starts. — Owner: ____
  - **Done when:** a 2×2 table (DenseNet121/ResNet50 × frozen/fine-tuned) with val AUC, sensitivity@spec, loss.
- [ ] **C4 [P1]** Fine-tuning ablation: unfrozen layers {0, 30, 60, all} and LR {1e-4, 3e-5, 1e-5}; keep BatchNorm layers in inference mode. — Owner: ____
- [ ] **C5 [P1]** Add a third architecture, EfficientNet-B0 (Keras EfficientNet has built-in rescaling and expects 0–255 inputs; its `preprocess_input` is a pass-through). — Owner: ____
- [ ] **C6 [P1]** Ablate horizontal flip and brightness/contrast on the best architecture. — Owner: ____
- [ ] **C7 [P2]** Aspect ratio: compare squash vs pad-to-square (`tf.image.resize_with_pad`). — Owner: ____
- [ ] **C8 [P1]** Train on a GPU (Colab/Kaggle) and record time per run in `results/metrics.csv`. — Owner: ____
- [ ] **C9 [P1]** Persist models safely: weights-only or `include_optimizer=False`; rebuild the architecture in code to reload (avoids the `Lambda` problem); host weights on Hugging Face/Drive and link from README; never commit files near 100 MB. — Owner: ____
- [ ] **C10 [P3]** Ensemble (average probabilities) of the best two models; choose the threshold on validation. — Owner: ____

### WS-D · Clinical-style evaluation (one shared evaluation module)

- [ ] **D1 [P0]** `src/evaluate.py`: given a model and a split, save `P(PNEUMONIA)` and labels (with file paths) to `results/preds/{model}_{split}.npz`. Every downstream analysis reads these files, so all models are evaluated by identical code. — Owner: ____
- [ ] **D2 [P0]** Validation metrics table for **all** models: accuracy, precision, sensitivity, specificity, F1, ROC-AUC, PR-AUC, log-loss, Brier. Save as `results/metrics.csv`. — Owner: ____
- [ ] **D3 [P0]** ROC and precision-recall curves, all models overlaid, on validation (and test for the final figure). Mark the chosen operating point. — Owner: ____
- [ ] **D4 [P0]** **Threshold selection on validation:** define the policy in advance (e.g. highest threshold with sensitivity ≥ 97%; also tabulate 90/95/97/99%), pick the final model **and** threshold, save to `results/threshold.json`, then freeze. Justify in clinical terms (missed child vs unnecessary follow-up). — Owner: ____
- [ ] **D5 [P0]** Evaluate on test **once** at the frozen threshold: confusion matrix (counts + row-normalised), classification report, bootstrap 95% CIs (2,000 resamples) for sensitivity, specificity, AUC. Also show the 0.5 result for reference. State how many times test was evaluated in total. — Owner: ____
- [ ] **D6 [P1]** Cost discussion with numbers: FN vs FP, plus a "per 1,000 children screened" table at prevalence 5%, 20%, and 62.6% (test) giving expected missed cases, false alarms, PPV, NPV. — Owner: ____
- [ ] **D7 [P2]** Paired comparison of the top two models on test: McNemar at the frozen thresholds and a bootstrap AUC difference. Say "cannot separate" when intervals overlap. — Owner: ____
- [ ] **D8 [P1]** Error analysis: gallery of **all** false negatives and a sample of false positives; break errors down by pneumonia subtype (bacterial vs viral from the filename), image size, brightness, markers present. Write what is common among them. — Owner: ____
- [ ] **D9 [P2]** Calibration: reliability diagram, ECE, Brier; temperature scaling fitted on validation, re-checked on test. (ResNet50's test loss of 0.45 vs DenseNet's 0.33 hints at overconfidence.) — Owner: ____

### WS-E · Explainability (Grad-CAM) and shortcut checks

- [ ] **E1 [P0]** Load models reliably: rebuild the architecture and `load_weights`, or replace the `Lambda` with a registered custom layer. Save the missing class-weighted baseline checkpoint. — Owner: ____
- [ ] **E2 [P0]** Grad-CAM that supports **nested backbones** (DenseNet121: last feature map; ResNet50: `conv5_block3_out`) and the scratch CNN; use pre-softmax logits (Appendix A.4). — Owner: ____
- [ ] **E3 [P0]** Fixed, documented example-selection protocol (seeded, not "first k"): 4 TP, 4 TN, **all or the 4 most confident FN**, 4 FP, plus 4 images with visible text markers; for the final model and for one scratch baseline as contrast. Run on validation while developing, on test for the final figures (no tuning from it). — Owner: ____
- [ ] **E4 [P1]** Quantify, don't just eyeball: fraction of heatmap mass inside an approximate lung region vs borders/corners/marker regions (central-region mask or simple thresholded lung mask), reported per class and for errors. — Owner: ____
- [ ] **E5 [P1]** Artefact probes: occlusion of markers/corners and the change in predicted probability; a "periphery-only" input test (lungs masked out) to see whether a model can classify from non-lung pixels. — Owner: ____
- [ ] **E6 [P2]** Sanity checks: Grad-CAM++ or occlusion sensitivity as a cross-check; model-parameter randomisation test (maps should change when weights are randomised). — Owner: ____
- [ ] **E7 [P1]** Written findings in the report: what the model attends to, where it looks wrong, and the limits of Grad-CAM (coarse 7×7 map, not proof of clinical reasoning). — Owner: ____

### WS-F · Repository, reproducibility, contributions

- [ ] **F1 [P0]** Create the structure from the brief: `README.md`, `requirements.txt`, `.gitignore`, `data/` (instructions only), `notebooks/01_eda`, `02_baseline_cnn`, `03_transfer_learning`, `04_evaluation_and_explainability`, `src/` (`dataset.py`, `models.py`, `train.py`, `evaluate.py`, `utils.py`), `results/`, `report/`. — Owner: ____
- [ ] **F2 [P0]** Move shared code (data loading, oversampling, augmentation, models, evaluation, Grad-CAM) into `src/`; notebooks import it; delete duplicated copies. — Owner: ____
- [ ] **F3 [P0]** `requirements.txt` with pinned versions (Python, TensorFlow/Keras, scikit-learn, numpy, matplotlib, Pillow, kagglehub); record versions in `results/env.txt`. — Owner: ____
- [ ] **F4 [P0]** `.gitignore`: `data/`, `chest_xray_cnn_splits/`, `saved_models/`, `*.keras`, `*.h5`, `*.pth`, `checkpoints/`, `.ipynb_checkpoints/`. Decide explicitly about the small baseline `.keras` files (1.2 MB each): commit or link. — Owner: ____
- [ ] **F5 [P0]** Root `README.md`: overview, dataset link + download steps (Kaggle credentials), setup, **exact run order**, results summary (generated from `results/metrics.csv`), key figures, limitations, **"not for clinical use" disclaimer**, **Contribution Table** (member, GitHub username, responsibilities, key files/experiments), AI-usage statement, citations. — Owner: ____
- [ ] **F6 [P1]** `results/`: `figures/`, `metrics.csv`, `preds/`, `threshold.json`, confusion matrices, `data_counts.csv`, `env.txt`. — Owner: ____
- [ ] **F7 [P1]** One `config` (seed, image size, batch size, paths, hyper-parameters); set Python/NumPy/TF seeds in one helper; use relative paths and a single data-location variable (no hard-coded user directories). — Owner: ____
- [ ] **F8 [P1]** Notebooks committed **with outputs** (restart-and-run-all before commit); confirm they render on GitHub. — Owner: ____
- [ ] **F9 [P0]** **Fresh-clone test** on a clean Colab runtime or new virtual environment: clone, install, download data, run. Add a smoke-test script that runs the whole pipeline on ~50 images in minutes. — Owner: ____
- [ ] **F10 [P0]** Git hygiene: all four members added as collaborators; each commits from their **own** account (check `git config user.email` matches their GitHub account); feature branches (`feature/data-split`, `exp/oversampling-fix`, `exp/transfer-learning`, `feature/gradcam`, …) and pull requests with a reviewer; commits spread over days; verify with `git shortlog -sne` and the GitHub contributors graph. — Owner: all
- [ ] **F11 [P1]** Model weights hosted outside Git (Hugging Face Hub or Drive) with links in README. — Owner: ____
- [ ] **F12 [P1]** Delete or rewrite `docs/IMBALANCE_INTEGRATION.md` (stale statements, nonexistent `imbalance.py`); merge useful parts into README. — Owner: ____
- [ ] **F13 [P2]** `LICENSE`, `CITATION.cff`, dataset licence/citation note (check the dataset's stated licence on Kaggle). — Owner: ____
- [ ] **F14 [P3]** GitHub Actions: lint + smoke test on every PR. — Owner: ____
- [ ] **F15 [P0]** AI-assistant disclosure: list where AI helped (code, docs, analysis) in README and report. Make sure every member can explain every line they own. — Owner: all

### WS-G · Project report (PDF, ~10–15 pages excluding appendix, outline from brief §8)

First page must carry: title, group number, members (leader marked), **public GitHub link**, **contribution table**.

- [ ] **G1 [P0]** Title page + contribution table + repo link. — Owner: ____
- [ ] **G2 [P1]** Introduction and motivation (pediatric pneumonia burden, radiologist scarcity, why accuracy is not enough). — Owner: ____
- [ ] **G3 [P1]** Related work, 3–5 papers in your own words (see Appendix C). — Owner: ____
- [ ] **G4 [P0]** Dataset: source, size, classes, official vs own splits, duplicate handling, EDA findings (imbalance ratios, shift, size/marker checks). — Owner: ____
- [ ] **G5 [P0]** Methodology: architecture **diagrams** (scratch CNN + transfer heads), loss, optimiser, hyper-parameters, augmentation rationale (incl. vertical-flip argument), imbalance strategies, threshold policy. — Owner: ____
- [ ] **G6 [P0]** Experiments and results: validation table for every model, learning curves, ROC/PR, ablations, multi-seed mean ± std, honest comparison including the oversampling bug story. — Owner: ____
- [ ] **G7 [P0]** Threshold + clinical evaluation: chosen operating point and why, test confusion matrix with CIs, FN-vs-FP cost discussion, "per 1,000 screened" table. — Owner: ____
- [ ] **G8 [P0]** Error analysis and explainability: Grad-CAM figures, misclassified samples, quantified attention, artefact probes. — Owner: ____
- [ ] **G9 [P0]** Limitations, ethics, future work: single-center pediatric data, label noise, dataset shift, patient-level leakage status, prevalence shift, automation bias, not a diagnostic tool, regulatory reality. — Owner: ____
- [ ] **G10 [P1]** Conclusion with the one-paragraph answer to "if this were used for screening, how many sick children would it miss?" — Owner: ____
- [ ] **G11 [P0]** References, AI-usage statement, test-set evaluation count. — Owner: ____
- [ ] **G12 [P1]** Abstract (150–200 words), written last. — Owner: ____
- [ ] **G13 [P1]** Consistency pass: every table/figure is generated from `results/`, captions and axis labels present, numbers identical in README and report, all figures referenced in text. Export PDF; verify the repo link opens while logged out. — Owner: ____

### WS-H · Beyond expectations (only after P0–P2)

- [ ] **H1 [P3]** Multi-seed (3–5) mean ± std for the final models plus bootstrap CIs everywhere (cheap, very high impact). — Owner: ____
- [ ] **H2 [P3]** Shortcut/artefact audit as a headline result: size/mode metadata baseline, periphery-only classifier, marker-occlusion test, adversarial validation (A4, E4, E5 combined into one section). — Owner: ____
- [ ] **H3 [P3]** Calibration + temperature scaling + **test-time augmentation** (brief's stretch goals). — Owner: ____
- [ ] **H4 [P3]** **Selective prediction / "refer to radiologist" band:** reject low-confidence cases and plot coverage vs sensitivity; frames the model as decision support. — Owner: ____
- [ ] **H5 [P3]** **Prevalence-aware operating-point analysis:** how PPV/NPV and the best threshold change from 62.6% (test) to realistic screening prevalence. — Owner: ____
- [ ] **H6 [P3]** Three-class problem NORMAL / BACTERIAL / VIRAL from filenames (brief stretch goal) with a patient-grouped split and per-class error analysis. — Owner: ____
- [ ] **H7 [P3]** Ensemble of DenseNet121 + ResNet50 (+ EfficientNet-B0) with validation-chosen threshold. — Owner: ____
- [ ] **H8 [P3]** **Gradio demo** (Hugging Face Spaces): upload X-ray → probability, chosen threshold, Grad-CAM overlay, bold **"Not for clinical use"** disclaimer. — Owner: ____
- [ ] **H9 [P3]** **Model card** (intended use, out-of-scope use, data, metrics with CIs, limitations) and a **datasheet** for the dataset. — Owner: ____
- [ ] **H10 [P3]** One-command reproduction (`make all` or `run_all.sh`), Dockerfile or `environment.yml`, CI badge in README. — Owner: ____
- [ ] **H11 [P3]** Preprocessing ablations: CLAHE, lung-region crop (simple heuristic or a pretrained segmenter if accessible), image size 224 vs 320. — Owner: ____
- [ ] **H12 [P3]** If any independent pediatric chest X-ray source can be legally accessed, run a zero-shot external check to quantify domain shift; otherwise discuss why external validation is missing. — Owner: ____

### WS-I · Team process and viva readiness

- [ ] **I1 [P0]** Day-one checklist: collaborators added, branches created, areas assigned, deadline from class announcement written at the top of this file, plan worked backwards from it. — Owner: leader
- [ ] **I2 [P1]** Cross-review: each member reviews at least one PR from another member. — Owner: all
- [ ] **I3 [P0]** Viva rehearsal (any member can be asked about any part). Each member should be able to answer:
  - Why is accuracy misleading here? What is 0.7394 / 0.6262?
  - How are the class weights computed (1,213 → 1.93, 3,475 → 0.67) and what do they do to the loss?
  - Why did oversampling collapse and how was it fixed?
  - What is data leakage here; how does patient-level splitting prevent it?
  - Why is BatchNorm kept in inference mode during fine-tuning?
  - What does sensitivity vs specificity mean for a screening tool; why was this threshold chosen?
  - How does Grad-CAM work; what can't it prove?
  - What does the model do on false negatives, and why?
  - Why are vertical flips a bad idea, and what about horizontal flips?
  - What would you change for real clinical use?

---

## 4. Suggested sequence

| Step | What | Depends on |
|---|---|---|
| 0 | Freeze test; fix ground rules; repo skeleton (F1, F3, F4); GPU environment; decide patient-level split (A2) | none |
| 1 | Manifest + split + EDA (A1–A5) | 0 |
| 2 | Shared `src/`, config, evaluation module (F2, F7, D1) | 1 |
| 3 | In parallel: scratch baselines + imbalance (B1–B8) and transfer learning (C1–C9) | 2 |
| 4 | Validation tables, ROC/PR, model + threshold choice and freeze (D2–D4) | 3 |
| 5 | Single test evaluation, cost analysis, error analysis (D5–D9) | 4 |
| 6 | Grad-CAM, quantification, artefact probes (E1–E7) | 4 (can start on val earlier) |
| 7 | Stretch items (WS-H) while writing | 5 |
| 8 | Report, README, fresh-clone test, viva rehearsal (WS-G, F5, F9, I3) | continuous; final pass after 5–6 |

Write report sections as results land; do not leave it all for the last day.

---

## 5. Submission gate — all must be ticked

**Minimum expectations from the brief**
- [ ] Scratch CNN plus ≥2 transfer models compared on the **same** split (validation numbers)
- [ ] ≥2 imbalance strategies compared with a recommendation (oversampling bug fixed)
- [ ] Justified threshold with ROC/PR curves and a clinical-style error discussion
- [ ] Grad-CAM analysis on correct **and** incorrect cases, including false negatives
- [ ] Report PDF with contribution table + public repo link on page 1
- [ ] Public GitHub repo with visible contributions from all four members

**Correctness**
- [ ] No test-based decisions after the freeze; test evaluation count disclosed
- [ ] Patient-level leakage addressed or explicitly listed as a limitation with evidence
- [ ] Frozen vs fine-tuned comparison present for both architectures
- [ ] Validation and test metrics include specificity and ROC-AUC, with confidence intervals on test

**Repository**
- [ ] Fresh clone runs end to end (or smoke test passes) on a clean environment
- [ ] `requirements.txt` pinned, seeds fixed, versions recorded
- [ ] Dataset and large weights not committed; download steps + weight links in README
- [ ] Notebook outputs saved and visible on GitHub; figures in `results/figures/`
- [ ] `git shortlog -sne` shows four authors with meaningful, spread-out commits
- [ ] Stale docs removed; no duplicated helper code

**Report and ethics**
- [ ] "Not a diagnostic tool" stated clearly (README and report)
- [ ] AI-assistant usage disclosed; all citations present
- [ ] Limitations: dataset bias, single center, label noise, shift, leakage, prevalence
- [ ] Every table/figure is traceable to a file in `results/`

---

## Appendix A · Code sketches (verify each before relying on it)

### A.1 Balanced oversampling that actually mixes classes

```python
AUTOTUNE = tf.data.AUTOTUNE
unbatched = train_ds.unbatch()                         # train_ds: label_mode="int"
normal_ds = unbatched.filter(lambda x, y: tf.equal(y, 0)).repeat()
pneu_ds   = unbatched.filter(lambda x, y: tf.equal(y, 1)).repeat()

balanced_ds = tf.data.Dataset.sample_from_datasets(
    [normal_ds, pneu_ds], weights=[0.5, 0.5], seed=SEED
).batch(BATCH_SIZE).prefetch(AUTOTUNE)

steps_per_epoch = (2 * n_pneumonia) // BATCH_SIZE      # same samples/epoch as before
model.fit(balanced_ds, steps_per_epoch=steps_per_epoch,
          validation_data=val_ds, epochs=EPOCHS)

# sanity check: positive share per batch should hover around 0.5
pos = [float(tf.reduce_mean(tf.cast(y, tf.float32))) for _, y in balanced_ds.take(50)]
print(min(pos), max(pos))
```
Apply augmentation **after** sampling so repeated minority images differ each time. Filtering twice decodes images twice; consider `.cache()` if memory allows.

### A.2 Patient-grouped split

```python
import re
from sklearn.model_selection import StratifiedGroupKFold

def group_id(path):
    n = path.name
    m = re.match(r"(person\d+)_", n)               # PNEUMONIA: person12_bacteria_45.jpeg
    if m: return m.group(1)
    m = re.match(r"(?:NORMAL2-)?(IM-\d+)", n)      # NORMAL: IM-0115-0001.jpeg (verify!)
    if m: return m.group(1)
    return n                                       # unknown pattern -> own group; count these

groups = [group_id(p) for p, _ in training_pool]
labels = [lbl for _, lbl in training_pool]
sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
train_idx, val_idx = next(sgkf.split(training_pool, labels, groups))
assert not ({groups[i] for i in train_idx} & {groups[i] for i in val_idx})
```
Check the regexes against real filenames first and report how many files fall into the fallback group.

### A.3 Threshold at a target sensitivity (validation only)

```python
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
fpr, tpr, thr = roc_curve(y_val, p_val)        # p_val = P(PNEUMONIA)
TARGET_SENS = 0.97
i = np.argmax(tpr >= TARGET_SENS)              # first index = highest threshold meeting target
threshold = float(thr[i])
print(threshold, "sens", tpr[i], "spec", 1 - fpr[i])
```

Bootstrap interval for any metric:

```python
rng = np.random.default_rng(SEED)
def boot_ci(y, p, thr, metric, B=2000):
    vals = []
    for _ in range(B):
        idx = rng.integers(0, len(y), len(y))
        vals.append(metric(y[idx], (p[idx] >= thr).astype(int)))
    return np.percentile(vals, [2.5, 97.5])
# sensitivity: recall_score(y, pred); specificity: recall_score(y, pred, pos_label=0)
```

### A.4 Grad-CAM for a transfer model with a nested backbone

```python
def gradcam_nested(model, backbone_name, preprocess_fn, img_0_255, class_idx):
    backbone = model.get_layer(backbone_name)             # e.g. "densenet121" / "resnet50"
    head = model.layers[-1]                               # final Dense(2, softmax)
    x = preprocess_fn(tf.cast(img_0_255[None], tf.float32))   # RGB, no augmentation
    with tf.GradientTape() as tape:
        feats = backbone(x, training=False)               # (1, 7, 7, C)
        tape.watch(feats)
        probs = head(tf.reduce_mean(feats, axis=(1, 2)))
        score = probs[:, class_idx]                       # better: use pre-softmax logits
    grads = tape.gradient(score, feats)
    w = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(w * feats, axis=-1))[0]
    cam = tf.image.resize(cam[..., None], (224, 224))[..., 0]
    return (cam / (tf.reduce_max(cam) + 1e-8)).numpy()
```
Load the model by rebuilding the architecture and calling `load_weights`, so the `Lambda` preprocessing layer is never deserialized.

---

## Appendix B · Figure and table inventory for `results/`

| File | Content |
|---|---|
| `figures/eda_counts.png`, `eda_intensity_per_class.png`, `eda_sizes.png`, `eda_markers.png` | EDA |
| `data_counts.csv` | counts per split/class, before/after de-duplication |
| `figures/curves_{model}.png` | train + val learning curves |
| `metrics.csv` | one row per model/split: acc, precision, sens, spec, F1, ROC-AUC, PR-AUC, loss, Brier |
| `ablation_*.csv` | baseline, fine-tuning, augmentation ablations (mean ± std over seeds) |
| `figures/roc_val.png`, `pr_val.png`, `roc_test.png` | curves, operating point marked |
| `threshold.json` | frozen model, policy, threshold |
| `figures/cm_test_{model}.png` | confusion matrices (counts + normalised) |
| `screening_table.csv` | per-1,000-screened outcomes at several prevalences |
| `figures/gradcam_*.png`, `gradcam_quant.csv` | explainability |
| `figures/errors_fn.png`, `errors_fp.png` | error galleries |
| `figures/reliability.png` | calibration |
| `env.txt` | library versions |

---

## Appendix C · Reading and citation candidates (verify every reference before citing)

- Kermany et al., 2018, *Cell* — source of this pediatric chest X-ray dataset (cohort description, labelling process).
- Rajpurkar et al., 2017 — CheXNet (DenseNet-121 for chest X-ray pneumonia detection).
- Selvaraju et al., 2017 — Grad-CAM.
- Huang et al., 2017 (DenseNet); He et al., 2016 (ResNet); Tan & Le, 2019 (EfficientNet).
- Lin et al., 2017 — focal loss.
- Guo et al., 2017 — *On Calibration of Modern Neural Networks*.
- Zech et al., 2018, *PLoS Medicine* — hospital-specific shortcuts in pneumonia detection from chest radiographs.
- DeGrave et al., 2021, *Nature Machine Intelligence* — chest X-ray models selecting shortcuts over signal.
- Geirhos et al., 2020 — *Shortcut Learning in Deep Neural Networks*.
- Adebayo et al., 2018 — *Sanity Checks for Saliency Maps*.
- Mitchell et al., 2019 — *Model Cards for Model Reporting*.
