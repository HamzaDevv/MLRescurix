# Kaggle Five-Problem Benchmark Specification & Metric Definitions

Updated: 2026-10-21 (Asia/Kolkata).
Manifest Reference: [BENCHMARK_SUITE.json](../BENCHMARK_SUITE.json).
Current Status: **Task specifications, split commitments, and metric definitions frozen.** Official dataset downloads pending Kaggle API credentials.

> [!IMPORTANT]
> Synthetic contract fixtures are provided for schema validation, pipeline wiring, and regression tests.
> **Synthetic cases do NOT count as authentic Kaggle benchmark results.**
> Official competition results require verified downloads of official competition datasets via Kaggle API with rule acceptance and pinned file hashes.

---

## 1. Task Definitions & Frozen Commitments

| Task ID | Competition | Target | Task Type | Primary Metric | Direction | Split Protocol | Baseline Pipeline |
|---|---|---|---|---|---|---|---|
| **K1** | Titanic | `Survived` | Binary Classification | `accuracy` | Maximize (↑) | Group-disjoint on frozen Family/Ticket mapping (5 dev folds, 20% final vault) | Fold-local Imputer + OneHot + Logistic Regression |
| **K2** | House Prices | `SalePrice` | Regression | `rmse_log_price` | Minimize (↓) | Shuffled 5-fold with separate final 20% holdout (positive price domain required) | Fold-local Imputer + OneHot + Ridge with log target transform |
| **K3** | Forest Cover Type | `Cover_Type` | 7-Class Multiclass | `accuracy` | Maximize (↑) | Stratified 5-fold with separate final 20% holdout (classes 1–7) | Fold-local HistGradientBoostingClassifier |
| **K4** | IEEE-CIS Fraud | `isFraud` | Imbalanced Binary | `roc_auc` | Maximize (↑) | Chronological rolling origin on `TransactionDT`, join on `TransactionID` | Bounded HistGradientBoostingClassifier with rare-event calibration |
| **K5** | Store Sales | `sales` | Panel Forecasting | `rmsle` | Minimize (↓) | Panel expanding origin on `date`, entities `(store_nbr, family)`, 15-day horizon | Seasonal Naive (period=7) with zero cold-start fallback |

---

## 2. Metric Formulations

### K1: Accuracy
$$\text{Accuracy} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(y_i = \hat{y}_i)$$

### K2: RMSE of Log Price
$$\text{RMSE}_{\text{log}} = \sqrt{\frac{1}{N} \sum_{i=1}^N (\ln(\hat{y}_i) - \ln(y_i))^2}$$
Predictions $\hat{y}_i$ must be strictly positive ($\hat{y}_i > 0$). Negative or zero predictions violate domain constraints.

### K3: Multiclass Accuracy
$$\text{Accuracy} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(y_i = \hat{y}_i), \quad y_i \in \{1, 2, 3, 4, 5, 6, 7\}$$

### K4: ROC-AUC
Area under the Receiver Operating Characteristic curve computed over out-of-fold predicted probabilities for class 1 (`isFraud=1`).

### K5: Root Mean Squared Logarithmic Error (RMSLE)
$$\text{RMSLE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (\ln(\hat{y}_i + 1) - \ln(y_i + 1))^2}$$
Sales values $y_i \ge 0$, predictions $\hat{y}_i \ge 0$.

---

## 3. Direction-Aware Gain Calculation (Argon A5 Corrected)
Let $d = +1$ for maximization (K1, K3, K4) and $d = -1$ for minimization (K2, K5):
$$\text{benefit} = d \times (\text{score}_{\text{new}} - \text{score}_{\text{old}})$$
$$\text{relative\_gain} = \frac{\text{benefit}}{\max(|\text{score}_{\text{old}}|, \epsilon)}$$
Promotion threshold: $\text{relative\_gain} \ge 0.005$ (+0.5%) on paired development assessment.
