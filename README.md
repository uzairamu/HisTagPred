# HisTagPred / TargetTrackHisTag

**TargetTrackHisTag: an experimental benchmark reveals a significant but modest sequence-encoded signal for His-tag placement preference**

Muhammad Uzair Ashraf, Muazzam, and Rizwan Hasan Khan  
Interdisciplinary Biotechnology Unit, Aligarh Muslim University, Aligarh, India

---

## Overview

This repository contains the dataset, results, and manuscript files for TargetTrackHisTag — the first machine-readable benchmark of experimentally derived bilateral His-tag placement outcomes, curated from the Northeast Structural Genomics Consortium (NESG) TargetTrack archive.

**Primary result:** Full-sequence ESM-2 (650M) embeddings with logistic regression achieved MCC = 0.168 (balanced accuracy = 0.583, permutation p = 0.0498, n = 156) against experimental labels — the first statistically significant sequence-based predictive signal for His-tag placement preference reported against an experimentally derived benchmark.

---

## Repository Structure

```
HisTagPred/
├── data/
│   ├── experimental_156_clean.csv     # Core benchmark dataset (156 proteins)
│   └── target_ids_156.csv             # TargetTrack target identifiers
├── results/
│   ├── permutation_156.csv            # 1,000-permutation null MCC distribution
│   ├── per_protein_predictions.csv    # Per-protein LOOCV predictions
│   ├── comparison_156.csv             # Method comparison summary
│   ├── fixed_window_analysis.csv      # Pre-specified length window results
│   ├── length_bin_methodB.csv         # Quartile length bin results
│   ├── sliding_window_methodB.csv     # Sliding window MCC analysis
│   ├── corrected_length_analysis.csv  # Median split analysis
│   └── extreme_bin_analysis.csv       # Extreme quartile analysis
├── audit_results/
│   ├── audit_a_tag_patterns_156.csv   # Tag pattern verification
│   ├── audit_b_confidence_156.csv     # Label confidence scores
│   ├── audit_c_properties_156.csv     # Per-protein physicochemical features
│   └── audit_c_results_156.csv        # Audit C statistical results
├── benchmark/
│   └── terminator_results.csv         # Terminator comparison (28 proteins)
├── manuscript/
│   ├── main.tex                       # LaTeX manuscript source
│   └── references.bib                 # Bibliography
├── figures/
│   ├── Figure1_confusion_matrix_v3.png
│   ├── Figure2_pipeline.png
│   ├── Figure3_comparison_v3.png
│   ├── Figure4_audits.png
│   └── Figure5_length_confidence.png
└── README.md
```

---

## Dataset

### experimental_156_clean.csv
The core TargetTrackHisTag benchmark. 156 proteins with experimentally recorded bilateral His-tag placement outcomes, curated from the NESG TargetTrack XML archive (97,418 His-tag construct entries → 804 bilateral candidates → 156 after alignment-based validation at ≥85% coverage).

| Column | Description |
|--------|-------------|
| target_id | NESG TargetTrack identifier |
| winner | Experimental preference: N or C |
| sequence_clean | Tag-stripped protein sequence |
| seq_len | Sequence length (aa) |
| n_score | Maximum pipeline progression score (N-terminal constructs) |
| c_score | Maximum pipeline progression score (C-terminal constructs) |
| coverage | Alignment coverage between N and C construct sequences |

**Class distribution:** 108 C-winners (69.2%), 48 N-winners (30.8%)

### ESM-2 Embeddings
Pre-computed ESM-2 (650M) full-sequence embeddings (`X_exp_156_esm.npy`, shape: 156×1280) and labels (`y_exp_156.npy`) are deposited at Zenodo (DOI: [ZENODO_DOI]) due to file size.

---

## Key Results

| Method | n | MCC | Balanced Acc | p (permutation) |
|--------|---|-----|--------------|-----------------|
| ESM-2 LogReg, full dataset (primary) | 156 | **0.168** | 0.583 | **0.0498** |
| ESM-2 LogReg, undersampled | 96 | 0.188 | 0.594 | 0.084 |
| Handcrafted terminal features | 62 | −0.065 | — | — |
| ESMFold pLDDT pseudo-labels (XGBoost) | 28 | −0.225 | — | — |
| ESMFold pLDDT drop (direct) | 27 | −0.243 | — | — |

---

## Reproducing the Primary Result

Analysis was performed in Python 3.10 on Google Colab (A100 GPU for ESM-2 embedding extraction, CPU for LOOCV).

**Dependencies:**
```bash
pip install fair-esm torch scikit-learn xgboost scipy numpy pandas matplotlib biopython
```

**Core pipeline (pseudocode):**
```python
import esm, torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import matthews_corrcoef

# 1. Load dataset
df = pd.read_csv('data/experimental_156_clean.csv')

# 2. Extract ESM-2 embeddings (or load from Zenodo)
# X = np.load('X_exp_156_esm.npy')  # from Zenodo
# y = np.load('y_exp_156.npy')

# 3. LOOCV evaluation
loo = LeaveOneOut()
y_true, y_pred = [], []
for train_idx, test_idx in loo.split(X):
    sc = StandardScaler()
    X_tr = sc.fit_transform(X[train_idx])
    X_te = sc.transform(X[test_idx])
    clf = LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000)
    clf.fit(X_tr, y[train_idx])
    y_true.append(y[test_idx][0])
    y_pred.append(clf.predict(X_te)[0])

mcc = matthews_corrcoef(y_true, y_pred)
# Expected: MCC ≈ 0.168
```

---

## Citation

If you use TargetTrackHisTag in your research, please cite:

```
Ashraf MU, Muazzam, Khan RH. TargetTrackHisTag: an experimental benchmark reveals
a significant but modest sequence-encoded signal for His-tag placement preference.
Bioinformatics Advances, 2026. [DOI to be added upon publication]
```

---

## Data Availability

- **This repository:** Dataset, audit results, LOOCV predictions, manuscript source
- **Zenodo (DOI: [ZENODO_DOI]):** ESM-2 embeddings, labels, full analysis outputs

---

## License

Dataset and results: CC BY 4.0  
Code: MIT

---

## Contact

Rizwan Hasan Khan — rhkhan.cb@amu.ac.in  
Interdisciplinary Biotechnology Unit, Aligarh Muslim University, Aligarh 202002, India
