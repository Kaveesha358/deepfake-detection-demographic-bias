# Results reported in the paper

All scores are 5-fold stratified cross-validation (`StratifiedKFold(shuffle=True, random_state=42)`). F1 uses Real (label 1) as the positive class. SD is the population SD over folds.

## Experiment A: balanced subset (N = 999; 499 real, 500 fake)

| Run | F1 base | F1 dual | p (F1) | AUC base | AUC dual | p (AUC) | Notes |
|---|---|---|---|---|---|---|---|
| 1 | 0.836 | 0.886 | 0.072 | 0.926 | 0.949 | 0.196 | different random 999-image sample, no seed |
| 2 | 0.838 | 0.879 | 0.102 | 0.930 | 0.945 | 0.003 | same sample as Run 3, no seed |
| 3 | 0.846 | 0.868 | n/a | 0.917 | 0.932 | n/a | seed 42; per-fold scores not preserved |
| Mean of runs | 0.840 | 0.878 | | 0.925 | 0.942 | | |

Per-fold differences (dual minus baseline):

| Fold | Run 1 ΔF1 | Run 1 ΔAUC | Run 2 ΔF1 | Run 2 ΔAUC |
|---|---|---|---|---|
| 1 | +0.018 | -0.004 | +0.022 | +0.019 |
| 2 | +0.040 | +0.040 | +0.021 | +0.009 |
| 3 | +0.068 | +0.049 | +0.022 | +0.020 |
| 4 | +0.003 | -0.021 | +0.119 | +0.018 |
| 5 | +0.118 | +0.051 | +0.021 | +0.010 |

### Run 3 pooled out-of-fold errors (499 real, 500 fake)

| Model | Real→Fake | Fake→Real | Real recall | Fake recall | Errors | Accuracy |
|---|---|---|---|---|---|---|
| Baseline | 94 | 54 | 0.812 | 0.892 | 148 | 85.2% |
| Dual-branch | 50 | 87 | 0.900 | 0.826 | 137 | 86.3% |

52 images are misclassified by both models, 96 only by the baseline and 85 only by the dual-branch model. Best-threshold F1 (tuned on the same data, exploratory): baseline 0.851 at 0.44, dual-branch 0.869 at 0.47.

## Experiment B: full filtered pool (N = 5,726; 2,980 real, 2,746 fake), Run 4

| Fold | F1 base | F1 dual | AUC base | AUC dual |
|---|---|---|---|---|
| 1 | 0.9496 | 0.9684 | 0.9884 | 0.9946 |
| 2 | 0.9480 | 0.9677 | 0.9896 | 0.9952 |
| 3 | 0.9520 | 0.9631 | 0.9879 | 0.9936 |
| 4 | 0.9544 | 0.9757 | 0.9905 | 0.9965 |
| 5 | 0.9567 | 0.9677 | 0.9930 | 0.9967 |
| Mean ± SD | 0.9521 ± 0.0031 | 0.9686 ± 0.0041 | 0.9899 ± 0.0018 | 0.9953 ± 0.0012 |

Paired t-test: F1 t(4) = 7.46, p = 0.0017; AUC-ROC t(4) = 12.34, p = 0.0002. Run 4 was not seeded and its predictions were not stored.

## Demographic filter check (100 real images, two annotators)

Annotator 1: 25 South Asian; Annotator 2: 16; both South Asian: 12; both not: 71; disagree: 17; raw agreement 83%; Cohen's κ = 0.485 (moderate).
