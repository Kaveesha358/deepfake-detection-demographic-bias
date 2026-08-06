import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from scipy.stats import ttest_rel

# Stratified 5-Fold Cross-Validation Setup
def get_stratified_cv(n_splits=5):
    
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Evaluation Metrics Calculation
ftl_score_metric = lambda y_true, y_pred_labels: f1_score(y_true, y_pred_labels)
auc_roc_metric = lambda y_true, y_pred_probs: roc_auc_score(y_true, y_pred_probs)

def evaluate_predictions(y_true, y_pred_probs, y_pred_labels):
   
    f1 = f1_score(y_true, y_pred_labels, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_pred_probs)
    except ValueError:
        auc = 0.5 
    return {"F1-Score": f1, "AUC-ROC": auc}

#  Paired t-test for Statistical Significance
def perform_paired_ttest(baseline_scores, dual_branch_scores):
   
    t_stat, p_value = ttest_rel(dual_branch_scores, baseline_scores)
    return {"t-statistic": t_stat, "p-value": p_value}

print("Evaluation pipeline functions defined successfully!")
