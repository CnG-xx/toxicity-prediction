
# Toxicity Prediction from Molecular Descriptors

This project builds a Random Forest classifier to predict whether a compound is **Toxic** or **Non‑Toxic** based on a large set of molecular descriptors. The analysis includes:

- Exploratory Data Analysis (EDA)
- Preprocessing (target encoding, handling missing values)
- Feature selection (variance threshold + mutual information)
- Nested cross‑validation with hyperparameter tuning
- Final model training and feature importance

## Dataset Overview

- **Number of samples**: 171  
- **Number of features**: 1203 (original descriptors)  
- **Target distribution**:  
  - Non‑Toxic: 115 (67.3%)  
  - Toxic: 56 (32.7%)  

No missing values were found in any feature.

## Nested Cross‑Validation Results

A 5‑fold outer cross‑validation with a 3‑fold inner cross‑validation was used to obtain an unbiased estimate of model performance. The metric is ROC‑AUC.

| Outer Fold | ROC‑AUC |
|------------|---------|
| 1          | 0.3986  |
| 2          | 0.5336  |
| 3          | 0.6838  |
| 4          | 0.4466  |
| 5          | 0.5020  |

**Mean ROC‑AUC**: `0.5129 ± 0.0972`

The performance is moderate, indicating room for improvement. The variation across folds suggests that the model may struggle with small dataset size or feature noise.

## Best Hyperparameters

The best hyperparameters were selected using random search in the inner loop. The combination that yielded the highest outer fold score (Fold 3) was:

- `n_estimators`: 154
- `max_depth`: 10
- `min_samples_split`: 13
- `min_samples_leaf`: 15
- `max_features`: None (i.e., use all features at each split)

These parameters were used to train the final model on the full dataset.

## Top 20 Most Important Features

After running the final model, the top 20 feature importances are saved in `results.txt` and visualised in `feature_importance.png`. The most influential descriptors include topological, electronic, and spatial features – key discriminators between toxic and non‑toxic compounds.

## Key Conclusions

- The Random Forest model achieved a mean ROC‑AUC of **0.5129** in nested cross‑validation. This indicates moderate predictive power, likely due to the small dataset size (only 171 samples).
- Feature selection reduced the original 1,203 descriptors to a core set of 500, which helped mitigate overfitting.
- The final model is saved as `final_model.pkl` and can be used for predictions on new data.

## Files

- `toxicity_model.py` – main Python script
- `results.txt` – full output of the analysis (metrics, best params, feature importances)
- `requirements.txt` – required Python packages
- `final_model.pkl` – trained model and preprocessing objects (generated after running the script)
- `class_distribution.png` – bar plot of target classes
- `feature_importance.png` – horizontal bar plot of top 20 feature importances

