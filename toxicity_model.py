
### **File 2: `toxicity_model.py`**


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from scipy.stats import randint
import joblib

RANDOM_STATE = 42

print("Loading data...")
df = pd.read_csv('ML2 dataset 1.csv')
print(f"Dataset shape: {df.shape}")
print("First 5 rows:")
print(df.head())

print("\nTarget distribution:")
print(df['Class'].value_counts())
sns.countplot(x='Class', data=df)
plt.title('Class Distribution')
plt.savefig('class_distribution.png')
plt.close()

missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing) > 0:
    print("\nColumns with missing values:")
    print(missing)
else:
    print("\nNo missing values found.")

le = LabelEncoder()
df['Class_encoded'] = le.fit_transform(df['Class'])   # Toxic=1, NonToxic=0

X = df.drop(columns=['Class', 'Class_encoded'])
y = df['Class_encoded']
feature_names = X.columns.tolist()
print(f"\nNumber of features: {X.shape[1]}")

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

outer_scores = []
best_params_list = []

print("\n" + "="*60)
print("Starting Nested Cross‑Validation (5 outer folds)")
print("="*60)

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
    print(f"\n--- Outer Fold {fold} ---")

    X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
    y_train_outer, y_test_outer = y.iloc[train_idx], y.iloc[test_idx]

    imputer = SimpleImputer(strategy='median')
    X_train_outer = imputer.fit_transform(X_train_outer)
    X_test_outer = imputer.transform(X_test_outer)

    var_selector = VarianceThreshold(threshold=0.01)
    X_train_outer = var_selector.fit_transform(X_train_outer)
    X_test_outer = var_selector.transform(X_test_outer)
    print(f"  After variance threshold: {X_train_outer.shape[1]} features")

    mi_selector = SelectKBest(score_func=mutual_info_classif, k=500)
    X_train_outer = mi_selector.fit_transform(X_train_outer, y_train_outer)
    X_test_outer = mi_selector.transform(X_test_outer)
    print(f"  After MI selection: {X_train_outer.shape[1]} features")

    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20),
        'max_features': ['sqrt', 'log2', None]
    }

    random_search = RandomizedSearchCV(
        rf, param_distributions=param_dist,
        n_iter=30, cv=inner_cv, scoring='roc_auc',
        random_state=RANDOM_STATE, n_jobs=-1, verbose=0
    )
    random_search.fit(X_train_outer, y_train_outer)

    best_params = random_search.best_params_
    best_params_list.append(best_params)
    print(f"  Best inner CV params: {best_params}")
    print(f"  Best inner CV ROC-AUC: {random_search.best_score_:.4f}")

    best_model = random_search.best_estimator_
    y_proba = best_model.predict_proba(X_test_outer)[:, 1]
    score = roc_auc_score(y_test_outer, y_proba)
    outer_scores.append(score)
    print(f"  Outer fold ROC-AUC: {score:.4f}")


print("\n" + "="*60)
print("Nested Cross‑Validation Results")
print("="*60)
print(f"Outer fold ROC-AUC scores: {outer_scores}")
print(f"Mean ROC-AUC: {np.mean(outer_scores):.4f} ± {np.std(outer_scores):.4f}")

print("\n" + "="*60)
print("Training final model on full dataset...")
print("="*60)

best_fold_idx = np.argmax(outer_scores)
best_params_final = best_params_list[best_fold_idx]
print(f"Using best params from outer fold {best_fold_idx+1}: {best_params_final}")

imputer_full = SimpleImputer(strategy='median')
X_full_imputed = imputer_full.fit_transform(X)

var_selector_full = VarianceThreshold(threshold=0.01)
X_full_highvar = var_selector_full.fit_transform(X_full_imputed)

mi_selector_full = SelectKBest(score_func=mutual_info_classif, k=500)
X_full_sel = mi_selector_full.fit_transform(X_full_highvar, y)

final_model = RandomForestClassifier(**best_params_final, random_state=RANDOM_STATE, n_jobs=-1)
final_model.fit(X_full_sel, y)

joblib.dump({
    'imputer': imputer_full,
    'variance_selector': var_selector_full,
    'mi_selector': mi_selector_full,
    'model': final_model,
    'label_encoder': le
}, 'final_model.pkl')
print("Final model saved as 'final_model.pkl'")


importances = final_model.feature_importances_
selected_features = feature_names[var_selector_full.get_support()][mi_selector_full.get_support()]
indices = np.argsort(importances)[::-1][:20]   # top 20

plt.figure(figsize=(10,6))
plt.title("Top 20 Feature Importances (Final Model)")
plt.barh(range(20), importances[indices][::-1], align='center')
plt.yticks(range(20), [selected_features[i] for i in indices][::-1])
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()
print("Feature importance plot saved as 'feature_importance.png'")


with open('results.txt', 'w') as f:
    f.write("TOXICITY PREDICTION RESULTS\n")
    f.write("===========================\n\n")
    f.write(f"Dataset shape: {df.shape}\n")
    f.write(f"Target distribution:\n{df['Class'].value_counts()}\n\n")
    f.write("Nested Cross‑Validation (5 outer folds)\n")
    f.write(f"Outer fold ROC-AUC scores: {outer_scores}\n")
    f.write(f"Mean ROC-AUC: {np.mean(outer_scores):.4f} ± {np.std(outer_scores):.4f}\n\n")
    f.write("Best hyperparameters per outer fold:\n")
    for i, params in enumerate(best_params_list, 1):
        f.write(f"  Fold {i}: {params}\n")
    f.write(f"\nFinal model uses best params from outer fold {best_fold_idx+1}\n")
    f.write("\nTop 20 Feature Importances:\n")
    for rank, idx in enumerate(indices, 1):
        f.write(f"{rank:2d}. {selected_features[idx]:30s} {importances[idx]:.4f}\n")

print("\nAll done! Results saved to 'results.txt'.")