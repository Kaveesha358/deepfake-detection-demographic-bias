# Hybrid Deepfake Detection with Biological Eye-Feature Analysis

This repository houses the formal research methodology, algorithmic preprocessing pipelines, and system architecture framework for **IT41043 — Intelligent Systems (Milestone 2)**. This project investigates the automated detection of authentic versus artificially generated (deepfake) facial imagery by implementing a carefully curated dataset subset alongside an advanced dual-branch detection framework.

---

- **Module:** IT41043 — Intelligent Systems | Horizon Campus, Faculty of Information Technology
- **Group:** ITBIN-2313-0020 (S.M.K. Sewwandi De Silva) & ITBIN-2313-0125 (H.C. Jayangi Wickramarathna)



## Project Overview

This project investigates whether a hybrid facial forgery detection model — combining conventional deep-learning-based feature extraction with biologically grounded eye-region analysis (corneal specular highlight symmetry and pupil-shape regularity) — can achieve both higher transparency and comparable or improved accuracy relative to standard "black box" CNN-based deepfake detectors, when applied to highly compressed, low-resolution GAN-generated images typical of South Asian social media platforms.

## Repository Structure

    annotation/
        annotation_results.md     - Cohen's Kappa results and interpretation
        annotation_sample.txt     - 100 filenames used for manual demographic annotation
    models/
        baseline_model.py         - SingleBranchBaseline (Branch A only)
        model.py                  - DualBranchDeepfakeDetector (proposed hybrid model)
    notebooks/
        deepfakedetect.ipynb      - Full pipeline: filtering, preprocessing, models, evaluation
    preprocessing/
        eye_extractor.py          - MTCNN-based eye-region extraction (Branch B input)
        filter_faces.py           - DeepFace-based demographic (Asian-subset) filtering
        pipeline.py                - Combined face-crop + compression + blur preprocessing
    evaluation.py                 - Stratified 5-fold CV, F1-score, AUC-ROC, paired t-test
    requirements.txt
    README.md


## Current Status (Milestone 2)

- [x] Research gap, question, and scope finalised (Milestone 1)
- [x] Dataset sourced and filtered (DeepFace demographic pre-filter on DeepDetect-2025 corpus, 500 real + 500 fake images)
- [x] Manual annotation completed — 100-image sample independently labelled by both researchers; Cohen's Kappa = 0.485 (moderate agreement)
- [x] Face detection & cropping implemented (MTCNN)
- [x] Compression simulation implemented (JPEG re-compression + Gaussian blur)
- [x] Eye-region extraction implemented (MTCNN keypoints — Branch B input)
- [x] Model architecture implemented and documented (dual-branch: EfficientNet-B0 + ResNet-18 fusion)
- [x] Baseline implemented (single-branch EfficientNet-B0)
- [x] Evaluation pipeline implemented (stratified 5-fold CV, F1-score, AUC-ROC, paired t-test)
- [ ] Model training and full-scale experimentation (Milestone 3)
- [ ] Results analysis and final report (Milestone 4)

## Dataset

Source pool: `ayushmandatta1/deepdetect-2025` (Kaggle). A demographic pre-filter using the DeepFace ethnicity classifier narrowed the pool to images classified under the "Asian" category, followed by manual review to confirm South Asian-adjacent relevance. See `annotation/annotation_results.md` for the full annotation methodology and inter-annotator agreement results.

## Getting Started

Clone the repository:

```bash
git clone https://github.com/Kaveesha358/deepfake-detection-demographic-bias.git
cd deepfake-detection-demographic-bias
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The full pipeline (dataset filtering → annotation → preprocessing → model definitions → evaluation setup) is implemented in `notebooks/deepfakedetect (2).ipynb`, developed and run on Kaggle Notebooks using the `ayushmandatta1/deepdetect-2025` dataset as input. To reproduce:

1. Open the notebook in Kaggle Notebooks (or Jupyter, with the source dataset available locally).
2. Run all cells in order — this performs demographic filtering, face detection/cropping, compression simulation, eye-region extraction, and defines the model, baseline, and evaluation functions.

## License

Academic project — Horizon Campus, IT41043, Academic Year 2026.
