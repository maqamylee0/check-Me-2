#!/usr/bin/env python
# coding: utf-8

# # 🩺 Check Me — Self-Screening Risk Triage: Training Notebook
# 

# #### Imports & Configuration

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
)
from sklearn.utils import class_weight
import xgboost as xgb

# sns.set_theme(style='whitegrid', palette='muted')
# plt.rcParams.update({'figure.dpi': 100, 'axes.titlesize': 13})

RANDOM_STATE = 42
DATA_PATH    = 'synthetic_breast_cancer_risk.csv'
MODEL_PATH   = 'model.joblib'

# Label mapping (used throughout)
LABEL_NAMES = {0: 'Green', 1: 'Yellow', 2: 'Red'}
LABEL_COLORS = {'Green': '#2ecc71', 'Yellow': '#f39c12', 'Red': '#e74c3c'}



# 
# #### Load Dataset

# In[2]:


df = pd.read_csv(DATA_PATH)
print(f'Shape: {df.shape}')
df.head()


# In[3]:


# Basic info & missing-value counts
df.info()
print('\nMissing values per column:')
print(df.isnull().sum()[df.isnull().sum() > 0])


# In[4]:


#plot ditstribution of classes
label_map = {0: 'Green', 1: 'Yellow', 2: 'Red'}  
counts = df['risk_level'].value_counts().sort_index()
try:
    counts.index = [label_map[i] for i in counts.index]
except KeyError:
    pass  
# label_map
counts


# In[5]:


#plot distribution of classes

ax = counts.plot(kind='bar', color=[LABEL_COLORS.get(l, '#999') for l in counts.index],
                 edgecolor='white', figsize=(6, 3))
ax.set_title('Risk-Level Class Distribution')
ax.set_xlabel('Risk Band')
ax.set_ylabel('Count')
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom', fontsize=11)
plt.tight_layout()
plt.show()


# The data seems to be heavily imbalanced, with most patients being assigned to the green category. We need to address this issue because it can lead to biased models that perform well on the majority class but fail to detect the minority class. hence we need to use a oversampling technique to balance the data. And probably use a different metric to evaluate the model like the F1 score and recall because precision is not a good metric for imbalanced data therefore we will not focus on accuracy values to evaluate the model.

# 
# #### Exploratory Data Analysis (EDA)

# In[ ]:


# For every value of risk level, how much does each numeric feature differ?
numeric_cols = ['family_history', 'age', 'lump_size_mm', 'symptom_duration_days', 'pregnancy_status', 'hormonal_contraception', 'previous_lumps', 'breast_pain', 'nipple_discharge', 'skin_dimples']
fig, axes = plt.subplots(1, len(numeric_cols), figsize=(15, 4))

for ax, col in zip(axes, numeric_cols):
    for lvl, grp in df.groupby('risk_level'):
        label = label_map.get(lvl, str(lvl))
        grp[col].dropna().plot.kde(ax=ax, label=label, color=LABEL_COLORS.get(label))
    ax.set_title(col)
    ax.legend()

plt.suptitle('Numeric Feature Distributions by Risk Level', y=1.02)
plt.tight_layout()
plt.show()


# This numeric feature distribution helps us identify features that we can use to separate the risk levels. for example old age and larger lamps are correlated with higher risk levels. Unlike symptom days which basically overlaps for all the risk levels, and hence shows that it is really not that useful hence we can remove it to save computational power.

# In[7]:


# ── Binary symptom prevalence per risk level ──────────────────────────────────
binary_cols = ['family_history', 'previous_lumps', 'breast_pain',
               'nipple_discharge', 'skin_dimples', 'pregnancy_status',
               'hormonal_contraception']

prevalence = (
    df.groupby('risk_level')[binary_cols]
      .mean()
      .rename(index=label_map)
      .T
)

prevalence.plot(kind='bar', figsize=(12, 4),
                color=[LABEL_COLORS.get(c, '#999') for c in prevalence.columns],
                edgecolor='white')
plt.title('Symptom Prevalence (mean) by Risk Level')
plt.ylabel('Proportion of patients')
plt.xticks(rotation=30, ha='right')
plt.legend(title='Risk Band')
plt.tight_layout()
plt.show()


# In[8]:


# ── Correlation heatmap (numeric + binary) ────────────────────────────────────
heat_cols = numeric_cols + binary_cols + ['risk_level']
corr = df[heat_cols].corr()

plt.figure(figsize=(10, 7))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            linewidths=0.5, square=True)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()


# In[9]:


# ── Categorical columns ───────────────────────────────────────────────────────
for cat_col in ['region', 'language']:
    if cat_col in df.columns:
        pd.crosstab(df[cat_col], df['risk_level'].map(label_map),
                    normalize='index').plot(kind='bar', stacked=True,
                                           color=[LABEL_COLORS[l] for l in ['Green', 'Yellow', 'Red']],
                                           figsize=(10, 3), edgecolor='white')
        plt.title(f'Risk Distribution by {cat_col.capitalize()}')
        plt.ylabel('Proportion')
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.show()


# ---
# ## 3 — Preprocessing Pipeline

# In[10]:


# ── Feature / target split ────────────────────────────────────────────────────
X = df.drop(columns=['risk_level'])
y = df['risk_level']

numeric_features     = ['age', 'lump_size_mm', 'symptom_duration_days']
categorical_features = ['region', 'language']
binary_features      = ['family_history', 'previous_lumps', 'breast_pain',
                         'nipple_discharge', 'skin_dimples',
                         'pregnancy_status', 'hormonal_contraception']

# ── Transformers ──────────────────────────────────────────────────────────────
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer,     numeric_features),
    ('cat', categorical_transformer, categorical_features),
    ('bin', 'passthrough',           binary_features)
])

# ── Train / test split (stratified) ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f'Train: {X_train.shape},  Test: {X_test.shape}')
print(f'Class counts (train): {y_train.value_counts().to_dict()}')


# In[11]:


# ── Sample weights for class balance ─────────────────────────────────────────
sample_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)
print('Sample weight stats:', {k: round(v, 3)
      for k, v in zip(['min', 'mean', 'max'],
                      [sample_weights.min(), sample_weights.mean(), sample_weights.max()])})


# ---
# ## 4 — Baseline Model: Logistic Regression

# In[12]:


# ──  TWEAK ME: Logistic Regression hyper-parameters ────────────────────────
LR_C          = 1.0         # Regularisation strength (smaller = stronger)
LR_SOLVER     = 'lbfgs'     # 'lbfgs', 'saga', 'newton-cg'
LR_MAX_ITER   = 1000

baseline_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        C=LR_C, solver=LR_SOLVER, max_iter=LR_MAX_ITER,
        class_weight='balanced', random_state=RANDOM_STATE
    ))
])

baseline_pipe.fit(X_train, y_train)
print('Baseline model trained ')


# In[13]:


# ── Cross-validation (optional sanity check) ──────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(baseline_pipe, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)
print(f'Logistic Regression 5-Fold F1 (weighted): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')


# ---
# ## 5 — Improved Model: XGBoost

# In[14]:


from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats

print('Starting AutoML hyperparameter tuning for XGBoost...')
# ──  AutoML: XGBoost hyper-parameters ────────────────────────────────────
base_xgb_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier',   xgb.XGBClassifier(
        objective        = 'multi:softprob',
        num_class        = 3,
        eval_metric      = 'mlogloss',
        random_state     = RANDOM_STATE,
        verbosity        = 0
    ))
])

param_dist = {
    'classifier__n_estimators': stats.randint(100, 500),
    'classifier__max_depth': stats.randint(3, 8),
    'classifier__learning_rate': stats.uniform(0.01, 0.19),
    'classifier__subsample': stats.uniform(0.6, 0.4),
    'classifier__colsample_bytree': stats.uniform(0.6, 0.4)
}

random_search = RandomizedSearchCV(
    estimator=base_xgb_pipe,
    param_distributions=param_dist,
    n_iter=10,
    scoring='f1_weighted',
    cv=3,
    verbose=1,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

random_search.fit(X_train, y_train, classifier__sample_weight=sample_weights)
print(f'Best parameters found: {random_search.best_params_}')
xgb_pipe = random_search.best_estimator_
print('XGBoost model trained via AutoML')


# ---
# ## 6 — Evaluation & Threshold Tuning

# In[15]:


def evaluate(model, X_test, y_test, name='Model'):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    print(f'\n{'='*50}')
    print(f'  {name}')
    print('='*50)
    print(classification_report(y_test, y_pred,
                                target_names=['Green', 'Yellow', 'Red']))

    roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
    print(f'ROC-AUC (weighted OvR): {roc_auc:.4f}')

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, y_pred),
        display_labels=['Green', 'Yellow', 'Red']
    ).plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'Confusion Matrix — {name}')
    plt.tight_layout()
    plt.show()

    return y_pred, y_prob

lr_pred, lr_prob   = evaluate(baseline_pipe, X_test, y_test, 'Logistic Regression (Balanced)')
xgb_pred, xgb_prob = evaluate(xgb_pipe,      X_test, y_test, 'XGBoost (Balanced)')


# In[16]:


# ── One-vs-Rest ROC curves ────────────────────────────────────────────────────
n_classes = 3
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

fig, axes = plt.subplots(1, n_classes, figsize=(15, 4))
for i, (cls_name, color) in enumerate(LABEL_COLORS.items()):
    for probs, model_name, ls in [
        (lr_prob,  'Logistic Regression', '--'),
        (xgb_prob, 'XGBoost',             '-')
    ]:
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], probs[:, i])
        auc_score   = auc(fpr, tpr)
        axes[i].plot(fpr, tpr, ls=ls, label=f'{model_name} (AUC={auc_score:.3f})')

    axes[i].plot([0, 1], [0, 1], 'k:', alpha=0.4)
    axes[i].set_title(f'ROC — {cls_name} (OvR)')
    axes[i].set_xlabel('FPR');  axes[i].set_ylabel('TPR')
    axes[i].legend(fontsize=8)

plt.suptitle('One-vs-Rest ROC Curves', y=1.01)
plt.tight_layout()
plt.show()


# In[17]:


# ──  TWEAK ME: Red-class threshold tuning ───────────────────────────────────
# Healthcare rule: maximise Recall for 'Red' (high risk).
# Lower the threshold → more Red predictions → higher Recall but lower Precision.

RED_IDX = 2  # class index for 'Red'

precisions, recalls, thresholds = precision_recall_curve(
    (y_test == RED_IDX).astype(int), xgb_prob[:, RED_IDX]
)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(thresholds, precisions[:-1], label='Precision (Red)', color='steelblue')
ax.plot(thresholds, recalls[:-1],    label='Recall (Red)',    color='tomato')
ax.axvline(x=0.35, color='gray', ls='--', alpha=0.7, label='Example threshold = 0.35')
ax.set_xlabel('Decision Threshold for Red')
ax.set_ylabel('Score')
ax.set_title('Precision-Recall vs Threshold for Red Class (XGBoost)')
ax.legend()
plt.tight_layout()
plt.show()


# In[18]:


# ── Apply custom thresholds & reassign risk bands ────────────────────────────
#  TWEAK these to adjust sensitivity/specificity:
RED_THRESHOLD    = 0.35   # P(Red) >= this → Red
YELLOW_THRESHOLD = 0.30   # P(Yellow) >= this (and not Red) → Yellow

def apply_thresholds(proba, red_thresh=RED_THRESHOLD, yellow_thresh=YELLOW_THRESHOLD):
    """Convert probability matrix to risk-band labels with custom thresholds."""
    labels = []
    for row in proba:
        if row[2] >= red_thresh:
            labels.append(2)   # Red
        elif row[1] >= yellow_thresh:
            labels.append(1)   # Yellow
        else:
            labels.append(0)   # Green
    return np.array(labels)

xgb_pred_custom = apply_thresholds(xgb_prob)
print('Custom-threshold classification report (XGBoost):')
print(classification_report(y_test, xgb_pred_custom,
                             target_names=['Green', 'Yellow', 'Red']))


# ---
# ## 7 — Interpretability: SHAP

# In[19]:


# ── Get feature names after preprocessing ────────────────────────────────────
fitted_preprocessor = xgb_pipe.named_steps['preprocessor']
ohe_cols   = fitted_preprocessor.named_transformers_['cat'] \
                                 .named_steps['onehot'] \
                                 .get_feature_names_out(categorical_features).tolist()
all_feature_names = numeric_features + ohe_cols + binary_features

# Transform training subset
X_train_t = fitted_preprocessor.transform(X_train.iloc[:500])

# SHAP explainer
explainer   = shap.Explainer(xgb_pipe.named_steps['classifier'])
shap_values = explainer(X_train_t)

print('SHAP values computed ')
print(f'SHAP output shape: {shap_values.values.shape}  (samples × features × classes)')


# In[20]:


# ── SHAP Summary Plot — Red class (most safety-critical) ─────────────────────
shap.summary_plot(
    shap_values.values[:, :, RED_IDX],   # SHAP values for Red
    X_train_t,
    feature_names=all_feature_names,
    show=True,
    max_display=10
)


# In[21]:


# ── Mean |SHAP| — top-5 global drivers for Red ───────────────────────────────
mean_shap = np.abs(shap_values.values[:, :, RED_IDX]).mean(axis=0)
importance_df = pd.DataFrame({'feature': all_feature_names, 'mean_shap': mean_shap})
importance_df = importance_df.sort_values('mean_shap', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(data=importance_df, y='feature', x='mean_shap', ax=ax, color='#e74c3c')
ax.set_title('Top 10 Features — Mean |SHAP| for Red Class')
ax.set_xlabel('Mean |SHAP value|')
plt.tight_layout()
plt.show()

print('\nTop 5 risk drivers (Red class):')
for i, row in importance_df.head(5).iterrows():
    print(f'  {row["feature"]}: {row["mean_shap"]:.4f}')


# In[22]:


# ── Logistic Regression Coefficients (baseline interpretability) ──────────────
lr_model = baseline_pipe.named_steps['classifier']
lr_preprocessor = baseline_pipe.named_steps['preprocessor']
ohe_cols_lr = lr_preprocessor.named_transformers_['cat'] \
                              .named_steps['onehot'] \
                              .get_feature_names_out(categorical_features).tolist()
lr_feature_names = numeric_features + ohe_cols_lr + binary_features

coef_df = pd.DataFrame(
    lr_model.coef_,
    index=['Green', 'Yellow', 'Red'],
    columns=lr_feature_names
).T

top_red = coef_df['Red'].abs().nlargest(10).index
coef_df.loc[top_red].plot(kind='bar', figsize=(12, 4), edgecolor='white')
plt.title('Logistic Regression Coefficients — Top 10 Features by |Red coeff|')
plt.axhline(0, color='black', lw=0.8)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()


# ---
# ## 8 — Save Model Artefacts

# In[23]:


# Save the XGBoost pipeline (best model)
joblib.dump(xgb_pipe, MODEL_PATH)
print(f'Model saved → {MODEL_PATH}')

# Save SHAP summary plot
plt.figure()
shap.summary_plot(shap_values.values[:, :, RED_IDX], X_train_t,
                  feature_names=all_feature_names, show=False)
plt.savefig('shap_summary.png', bbox_inches='tight')
plt.close()
print('SHAP plot saved → shap_summary.png')


# ---
# ## 9 — Quick Prediction Demo
# 
# Manually craft a sample patient and run it through the pipeline.

# In[24]:


# ──  TWEAK ME: modify patient inputs ───────────────────────────────────────
sample_patient = pd.DataFrame([{
    'age':                    45,
    'family_history':          1,
    'previous_lumps':          1,
    'breast_pain':             0,
    'nipple_discharge':        1,
    'skin_dimples':            0,
    'lump_size_mm':           15.0,
    'symptom_duration_days':  30,
    'pregnancy_status':        0,
    'hormonal_contraception':  1,
    'region':                 'East Africa',
    'language':               'Swahili'
}])

proba  = xgb_pipe.predict_proba(sample_patient)[0]
custom = apply_thresholds(proba.reshape(1, -1))[0]

BAND = {0: ' Green', 1: ' Yellow', 2: ' Red'}
print(f'Class probabilities:  Green={proba[0]:.3f}, Yellow={proba[1]:.3f}, Red={proba[2]:.3f}')
print(f'Risk Band:            {BAND[custom]}')

# Simple recommendation lookup
RECOMMENDATIONS = {
    0: [' Perform monthly self-check', ' Maintain healthy diet & weight',
        ' Stay active (150 min/week)', ' Annual screening reminder'],
    1: ['⏱ Track symptoms for 7–14 days',
        ' Consider booking a screening if symptoms persist',
        ' Prioritise sleep and stress management',
        ' Keep a symptom diary'],
    2: [' **Strongly recommended**: visit a clinic for screening',
        ' Contact your healthcare provider this week',
        ' Bring this summary to your appointment',
        ' This is not a diagnosis — a professional assessment is needed']
}

print('\nRecommendations:')
for r in RECOMMENDATIONS[custom]:
    print(f'  {r}')

print('\n  Disclaimer: This is NOT a diagnosis. Consult a healthcare professional for any concerns.')

