#========================================================================
 # IMPORTS
#========================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

#========================================================================
 # SECTION 1 -- Data Loading & Exploration (EDA)
#========================================================================
df = sns.load_dataset('titanic')
print('Shape:', df.shape)
print(df.head())

df.info()
print(df.describe())

missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'count': missing, 'pct': missing_pct})
print(missing_df[missing_df['count'] > 0])

print('Survived value counts:')
print(df['survived'].value_counts())
print('\nClass balance (%):')
print(df['survived'].value_counts(normalize=True) * 100)

# fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# df['survived'].value_counts().plot(kind='bar', ax=axes[0], color=['salmon', 'steelblue'])
# axes[0].set_title('Survival Count')
# axes[0].set_xticklabels(['Died (0)', 'Survived (1)'], rotation=0)
# df['age'].dropna().hist(bins=30, ax=axes[1], color='steelblue', edgecolor='black')
# axes[1].set_title('Age Distribution')
# df.groupby('pclass')['survived'].mean().plot(kind='bar', ax=axes[2], color='steelblue')
# axes[2].set_title('Survival Rate by Class')
# axes[2].set_xticklabels(['1st', '2nd', '3rd'], rotation=0)
# plt.tight_layout()
# plt.show()

# numeric_cols = df.select_dtypes(include=[np.number]).columns
# plt.figure(figsize=(8, 6))
# sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm')
# plt.title('Correlation Heatmap')
# plt.show()

#========================================================================
 # SECTION 2 -- Preprocessing & Feature Engineering
#========================================================================
data = df.copy()

# Drop redundant/duplicate columns
cols_to_drop = ['alive', 'embark_town', 'who', 'adult_male', 'alone', 'deck', 'class']
data.drop(columns=cols_to_drop, inplace=True)
print('Columns after dropping:', list(data.columns))

# Handle missing values
data['age']      = data['age'].fillna(data['age'].median())
data['embarked'] = data['embarked'].fillna(data['embarked'].mode()[0])
print('Missing values after imputation:')
print(data.isnull().sum())

# Feature engineering
data['family_size']    = data['sibsp'] + data['parch'] + 1
data['is_alone']       = (data['family_size'] == 1).astype(int)
data['age_bin']        = pd.cut(data['age'], bins=[0, 12, 60, 100],
                                labels=['child', 'adult', 'senior'])
data['fare_per_person'] = data['fare'] / data['family_size']

print('Correlation of new features with survived:')
print(data[['family_size', 'is_alone', 'fare_per_person', 'survived']].corr()['survived'])

# Encode categoricals
data['sex'] = LabelEncoder().fit_transform(data['sex'])
data = pd.get_dummies(data, columns=['embarked', 'age_bin'], drop_first=True)
print('Columns after encoding:', list(data.columns))

# Split into features and target
X_fe = data.drop('survived', axis=1)
X_fe = X_fe.fillna(X_fe.median(numeric_only=True))  # catch any NaN from get_dummies on Categorical
y_fe = data['survived']

# Train/test split (stratified to preserve class balance)
X_train_fe, X_test_fe, y_train_fe, y_test_fe = train_test_split(
    X_fe, y_fe, test_size=0.2, random_state=42, stratify=y_fe
)
print(f'Train: {X_train_fe.shape}  |  Test: {X_test_fe.shape}')
print('Train class balance:', y_train_fe.value_counts(normalize=True).to_dict())

# Scale -- required for k-NN, LR, SVM; fit only on train
scaler_fe = StandardScaler()
X_train_fe_scaled = scaler_fe.fit_transform(X_train_fe)
X_test_fe_scaled  = scaler_fe.transform(X_test_fe)
print('Feature set:', list(X_fe.columns))

#========================================================================
 # SECTION 4 -- Model Selection & Training
#========================================================================
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree':       DecisionTreeClassifier(random_state=42),
    'k-NN (k=5)':         KNeighborsClassifier(n_neighbors=5),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM':                 SVC(probability=True),
}

results = {}

for name, model in models.items():
    if name in ['Decision Tree', 'Random Forest']:
        model.fit(X_train_fe, y_train_fe)
        y_pred = model.predict(X_test_fe)
        y_prob = model.predict_proba(X_test_fe)[:, 1]
        cv_scores = cross_val_score(model, X_fe, y_fe, cv=5, scoring='accuracy')
    else:
        model.fit(X_train_fe_scaled, y_train_fe)
        y_pred = model.predict(X_test_fe_scaled)
        y_prob = model.predict_proba(X_test_fe_scaled)[:, 1]
        cv_scores = cross_val_score(model, X_train_fe_scaled, y_train_fe, cv=5, scoring='accuracy')

    results[name] = {
        'accuracy':  accuracy_score(y_test_fe, y_pred),
        'precision': precision_score(y_test_fe, y_pred),
        'recall':    recall_score(y_test_fe, y_pred),
        'f1':        f1_score(y_test_fe, y_pred),
        'roc_auc':   roc_auc_score(y_test_fe, y_prob),
        'cv_mean':   cv_scores.mean(),
        'cv_std':    cv_scores.std(),
        'y_pred':    y_pred,
        'y_prob':    y_prob,
    }

print('Training complete.')

summary = pd.DataFrame({
    name: {
        'Accuracy':  f"{r['accuracy']:.3f}",
        'Precision': f"{r['precision']:.3f}",
        'Recall':    f"{r['recall']:.3f}",
        'F1':        f"{r['f1']:.3f}",
        'ROC-AUC':   f"{r['roc_auc']:.3f}",
        'CV Mean':   f"{r['cv_mean']:.3f} +/- {r['cv_std']:.3f}",
    }
    for name, r in results.items()
}).T
print(summary)

metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
compare_df = pd.DataFrame(
    {name: {m: r[m] for m in metrics} for name, r in results.items()}
).T
# compare_df.plot(kind='bar', figsize=(12, 5), ylim=(0.5, 1.0))
# plt.title('Model Comparison')
# plt.xticks(rotation=20, ha='right')
# plt.legend(loc='lower right')
# plt.tight_layout()
# plt.show()

# dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
# dt_model.fit(X_train_fe, y_train_fe)
# plt.figure(figsize=(16, 6))
# plot_tree(dt_model, feature_names=X_train_fe.columns.tolist(),
#           class_names=['Died', 'Survived'], filled=True, rounded=True, fontsize=9)
# plt.title('Decision Tree (max_depth=3)')
# plt.show()

k_range  = range(1, 21)
k_scores = []
for k in k_range:
    knn   = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X_train_fe_scaled, y_train_fe, cv=5, scoring='accuracy').mean()
    k_scores.append(score)

# plt.figure(figsize=(8, 4))
# plt.plot(k_range, k_scores, marker='o')
# plt.xlabel('k (number of neighbors)')
# plt.ylabel('CV Accuracy')
# plt.title('k-NN: Accuracy vs k')
# plt.xticks(k_range)
# plt.grid(True)
# plt.show()

best_k = k_range[k_scores.index(max(k_scores))]
print(f'Best k: {best_k}  |  CV Accuracy: {max(k_scores):.3f}')

#========================================================================
 # SECTION 5 -- Evaluation Metrics
#========================================================================
best_name = max(results, key=lambda k: results[k]['f1'])
best_res  = results[best_name]
print(f'Best model: {best_name}')

cm = confusion_matrix(y_test_fe, best_res['y_pred'])
# plt.figure(figsize=(5, 4))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=['Died', 'Survived'], yticklabels=['Died', 'Survived'])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title(f'Confusion Matrix -- {best_name}')
# plt.show()

tn, fp, fn, tp = cm.ravel()
print(f'TN={tn}  FP={fp}  FN={fn}  TP={tp}')
print(f'Precision = TP/(TP+FP) = {tp/(tp+fp):.3f}')
print(f'Recall    = TP/(TP+FN) = {tp/(tp+fn):.3f}')
print(classification_report(y_test_fe, best_res['y_pred'], target_names=['Died', 'Survived']))

# plt.figure(figsize=(8, 6))
# for name, r in results.items():
#     fpr, tpr, _ = roc_curve(y_test_fe, r['y_prob'])
#     plt.plot(fpr, tpr, label=f"{name} (AUC={r['roc_auc']:.3f})")
# plt.plot([0, 1], [0, 1], 'k--', label='Random')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curves')
# plt.legend()
# plt.grid(True)
# plt.show()

# rf_model    = models['Random Forest']
# importances = pd.Series(rf_model.feature_importances_, index=X_train_fe.columns)
# importances.sort_values(ascending=True).plot(kind='barh', figsize=(8, 6))
# plt.title('Feature Importances -- Random Forest')
# plt.tight_layout()
# plt.show()

#========================================================================
 # SECTION 6 -- Hyperparameter Tuning (GridSearchCV)
#========================================================================
# --- Decision Tree ---
param_grid_dt = {
    'max_depth':         [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'criterion':         ['gini', 'entropy'],
}
grid_dt = GridSearchCV(DecisionTreeClassifier(random_state=42),
                       param_grid_dt, cv=5, scoring='f1', n_jobs=-1)
grid_dt.fit(X_train_fe, y_train_fe)
print('Best DT params:', grid_dt.best_params_)
print('Best CV F1:    ', round(grid_dt.best_score_, 3))

# --- k-NN ---
param_grid_knn = {
    'n_neighbors': list(range(1, 21)),
    'weights':     ['uniform', 'distance'],
    'metric':      ['euclidean', 'manhattan'],
}
grid_knn = GridSearchCV(KNeighborsClassifier(),
                        param_grid_knn, cv=5, scoring='f1', n_jobs=-1)
grid_knn.fit(X_train_fe_scaled, y_train_fe)
print('Best k-NN params:', grid_knn.best_params_)
print('Best CV F1:       ', round(grid_knn.best_score_, 3))

# --- Logistic Regression ---
param_grid_lr = {
    'C':       [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver':  ['liblinear'],
}
grid_lr = GridSearchCV(LogisticRegression(max_iter=1000),
                       param_grid_lr, cv=5, scoring='f1', n_jobs=-1)
grid_lr.fit(X_train_fe_scaled, y_train_fe)
print('Best LR params:', grid_lr.best_params_)
print('Best CV F1:    ', round(grid_lr.best_score_, 3))

# --- Random Forest ---
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth':    [3, 5, 7, None],
    'max_features': ['sqrt', 'log2'],
}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42),
                       param_grid_rf, cv=5, scoring='f1', n_jobs=-1)
grid_rf.fit(X_train_fe, y_train_fe)
print('Best RF params:', grid_rf.best_params_)
print('Best CV F1:    ', round(grid_rf.best_score_, 3))

# --- SVM ---
param_grid_svm = {
    'C':      [0.1, 1, 10],
    'kernel': ['rbf', 'linear'],
    'gamma':  ['scale', 'auto'],
}
grid_svm = GridSearchCV(SVC(probability=True),
                        param_grid_svm, cv=5, scoring='f1', n_jobs=-1)
grid_svm.fit(X_train_fe_scaled, y_train_fe)
print('Best SVM params:', grid_svm.best_params_)
print('Best CV F1:     ', round(grid_svm.best_score_, 3))

# --- Evaluate all tuned models ---
tuned_models = {
    'Tuned Decision Tree':       (grid_dt.best_estimator_,  X_test_fe),
    'Tuned k-NN':                (grid_knn.best_estimator_, X_test_fe_scaled),
    'Tuned Logistic Regression': (grid_lr.best_estimator_,  X_test_fe_scaled),
    'Tuned Random Forest':       (grid_rf.best_estimator_,  X_test_fe),
    'Tuned SVM':                 (grid_svm.best_estimator_, X_test_fe_scaled),
}
tuned_results = {}
for name, (model, X_test_input) in tuned_models.items():
    y_pred = model.predict(X_test_input)
    print(f'\n{name}')
    print(f'  Accuracy : {accuracy_score(y_test_fe, y_pred):.3f}')
    print(f'  F1 Score : {f1_score(y_test_fe, y_pred):.3f}')
    print(f'  ROC-AUC  : {roc_auc_score(y_test_fe, model.predict_proba(X_test_input)[:, 1]):.3f}')
    tuned_results[name] = {'f1': f1_score(y_test_fe, y_pred), 'y_pred': y_pred}

#========================================================================
 # FINAL -- Select Best Model & Save Predictions
#========================================================================
best_tuned_name = max(tuned_results, key=lambda k: tuned_results[k]['f1'])
best_tuned_pred = tuned_results[best_tuned_name]['y_pred']

print(f'\nSelected model: {best_tuned_name}  (F1 = {tuned_results[best_tuned_name]["f1"]:.3f})')

output = pd.DataFrame({
    'actual':    y_test_fe.values,
    'predicted': best_tuned_pred,
})
output.to_csv('predictions.csv', index=False)
print('Predictions saved to predictions.csv')

