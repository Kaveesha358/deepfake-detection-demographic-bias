# Hybrid Deepfake Detection with Eye-Region Features

This repository houses the code, preprocessing pipelines, experiment logs and results for **IT41043 — Intelligent Systems**. The project investigates whether adding an eye-region branch to a standard CNN improves the detection of GAN-generated facial imagery when images are partly degraded by simulated social-media compression.

**Paper:** *A Hybrid CNN and Eye-Region Detector for GAN-Generated Faces Under Simulated Compression: A Study Motivated by Sri Lankan Social Media* (see [`paper/`](paper/)).

---

- **Module:** IT41043 — Intelligent Systems | Horizon Campus, Faculty of Information Technology
- **Group:** ITBIN-2313-0020 (S.M.K. Sewwandi De Silva) & ITBIN-2313-0125 (H.C. Jayangi Wickramarathna)

## Project Overview

We compare two models trained under identical conditions:

- **Baseline:** EfficientNet-B0 on the cropped face (single branch).
- **Dual-branch model (proposed):** EfficientNet-B0 on the whole face + ResNet-18 on an eye-region crop. The two feature vectors are concatenated (1792-d), passed through a 256-unit layer (ReLU, dropout 0.5) and a sigmoid classifier.

Both models are ImageNet-pretrained and trained with Adam (learning rate 1e-4), binary cross-entropy, batch size 16 and 10 epochs per fold. The dual-branch model also contains an "explanation" head; it is implemented but **not trained or evaluated**, so this repository makes no interpretability claim.

### Main results (all numbers in [`results/RESULTS.md`](results/RESULTS.md))

| Experiment | Images | Runs | F1 base → dual | AUC-ROC base → dual |
|---|---|---|---|---|
| A (balanced subset) | 999 | 3 | 0.840 → 0.878 (mean of runs) | 0.925 → 0.942 (mean of runs) |
| B (full filtered pool) | 5,726 | 1 | 0.952 → 0.969 (p = 0.0017) | 0.990 → 0.995 (p = 0.0002) |

In Experiment A the dual-branch model is ahead on average in every run, but the advantage is small and its significance varies between runs. In Experiment B it is ahead in all ten fold-wise comparisons (5 folds × 2 metrics). In Experiment A (Run 3) the dual-branch model rejects fewer real faces (50 vs. 94) but passes more fake faces as real (87 vs. 54).

### Important limitations (please read before reusing the results)

- **Eye crops are not degraded.** The eye crop is cut from the original image, so the compression and blur applied to the face crop never reach the eye branch. The dual-branch model therefore sees cleaner pixels than the baseline, which confounds the comparison.
- **Only about half of the face crops are compressed** (JPEG with probability 0.5, Gaussian blur with probability 0.3). The per-image choice was not stored.
- **"Asian-classified", not verified South Asian.** The corpus was filtered with the DeepFace ethnicity classifier. In a 100-image human check (real images only), both annotators judged only 12 images to be South Asian (Cohen's κ = 0.485, moderate agreement).
- **No manual review** was applied to the training images. Manual labelling was used only for the 100-image agreement check.
- The dual-branch model has an extra fusion layer that the baseline lacks.
- For Experiment B only the final scores were kept (no stored predictions, no error analysis). Error analysis exists for Experiment A, Run 3 only.
- Images where no face was detected were excluded (1 real image in Experiment A, 11 images in Experiment B).

## Repository Structure

```
annotation/
    annotation_results.md     - Cohen's Kappa results and interpretation
    annotation_sample.txt     - 100 filenames used for manual demographic annotation
models/
    baseline_model.py         - SingleBranchBaseline (Branch A only)
    model.py                  - DualBranchDeepfakeDetector (proposed hybrid model)
notebooks/
    deepfake_detection_pipeline.ipynb - Final notebook: filtering, preprocessing, models,
                                        5-fold CV, error analysis (Experiment A, Runs 2-3)
    run1_original.ipynb       - Notebook with the outputs of Run 1 (Experiment A)
paper/
    (final paper PDF)
preprocessing/
    eye_extractor.py          - MTCNN-based eye-region extraction (Branch B input)
    filter_faces.py           - DeepFace-based demographic (Asian-subset) filtering
    pipeline.py               - Combined face-crop + compression + blur preprocessing
results/
    RESULTS.md                - All numbers reported in the paper
    run4_full_pool_log.txt    - Console output of Experiment B (Run 4)
evaluation.py                 - Stratified 5-fold CV, F1-score, AUC-ROC, paired t-test
requirements.txt
README.md
```

The notebook `notebooks/deepfake_detection_pipeline.ipynb` is the authoritative implementation; the `.py` files are earlier modular versions of the same components.

## Current Status (All milestones completed)

- [x] Research gap, question, and scope finalised (Milestone 1)
- [x] Dataset sourced and filtered (DeepFace demographic pre-filter on the DeepDetect-2025 corpus)
- [x] Annotation check completed — 100-image sample independently labelled by both researchers; Cohen's Kappa = 0.485 (moderate agreement)
- [x] Face detection & cropping implemented (MTCNN)
- [x] Compression simulation implemented (JPEG re-compression + Gaussian blur, applied to the face crop)
- [x] Eye-region extraction implemented (MTCNN keypoints — Branch B input)
- [x] Model architecture implemented (dual-branch: EfficientNet-B0 + ResNet-18 fusion)
- [x] Baseline implemented (single-branch EfficientNet-B0)
- [x] Evaluation pipeline implemented (stratified 5-fold CV, F1-score, AUC-ROC, paired t-test)
- [x] Model training and full-scale experimentation (Milestone 3) — Experiment A: 3 runs on 999 images; Experiment B: 1 run on 5,726 images
- [x] Results analysis and final report (Milestone 4) — see `paper/` and `results/`

## Dataset

Source pool: `ayushmandatta1/deepdetect-2025` (Kaggle). The images are **not** included in this repository; obtain them from Kaggle and respect the original licences.

1. The DeepFace ethnicity classifier is applied to the real and fake pools, keeping images whose dominant race label is "asian" (2,986 real and 2,751 fake images).
2. **Experiment A:** the source folder is shuffled (no fixed seed) and the first 500 images per class classified as "asian" are kept (999 remain after face detection: 499 real, 500 fake).
3. **Experiment B:** all images classified as "asian" are used, with the per-class cap removed (5,726 remain after face detection: 2,980 real, 2,746 fake).

The corpus is therefore **Asian-classified, not verified South Asian**. See `annotation/annotation_results.md` for the annotation check: Annotator 1 labelled 25 of 100 images as South Asian, Annotator 2 labelled 16, both agreed on 12 and both rejected 71 (raw agreement 83%, κ = 0.485).

## Getting Started

Clone the repository:

```
git clone https://github.com/Kaveesha358/deepfake-detection-demographic-bias.git
cd deepfake-detection-demographic-bias
```

Install dependencies:

```
pip install -r requirements.txt
```

The full pipeline (dataset filtering → annotation → preprocessing → models → 5-fold cross-validation → error analysis) is implemented in `notebooks/deepfake_detection_pipeline.ipynb`, developed and run on **Kaggle Notebooks** (Python 3.12, NVIDIA Tesla T4 GPU) with `ayushmandatta1/deepdetect-2025` as input. Paths such as `/kaggle/input/...` and `/kaggle/working/...` are hard-coded. To reproduce:

1. Create a Kaggle notebook, attach the `ayushmandatta1/deepdetect-2025` dataset and enable a GPU.
2. Upload `notebooks/deepfake_detection_pipeline.ipynb` and run all cells in order.
3. **Install order matters:** if MTCNN fails with "LZ4 is not installed" although `lz4` is installed, restart the session, run `pip install lz4 mtcnn` first, and only then run the imports.

### Reproducibility notes

- The sampling step of Experiment A uses `random.shuffle` **without a seed**, so every fresh run draws a different 999-image sample. Runs 2 and 3 share one sample; Run 1 used another.
- Folds use `StratifiedKFold(shuffle=True, random_state=42)`, so they are identical whenever the data are identical.
- Only Run 3 set a random seed (42) for model initialisation. GPU non-determinism can still change the numbers slightly.
- `requirements.txt` lists the packages used; versions that are not pinned there were not recorded at the time of the runs.
- Trained model weights are **not included**: models are re-trained in every fold and no final checkpoint was saved.



## Ethics

All images come from public academic datasets and no new personal data were collected. The filter is a weak proxy for South Asian appearance, error rates on subgroups were not measured, and the results must not be used as evidence about individual images without human review.

## License

Academic project — Horizon Campus, IT41043, Academic Year 2026. The dataset and its images remain under their original licences.
