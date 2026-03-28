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
train = pd.read_csv('datasets/titanic/train.csv')
test  = pd.read_csv('datasets/titanic/test.csv')
df = train  # EDA runs on train only

print('Train shape:', train.shape)
print('Test shape: ', test.shape)
print(df.head())

df.info()
print(df.describe())

missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'count': missing, 'pct': missing_pct})
print(missing_df[missing_df['count'] > 0])

print('Survived value counts:')
print(df['Survived'].value_counts())
print('Survived balance (%):')
print(df['Survived'].value_counts(normalize=True) * 100)

#========================================================================
 # SECTION 2 -- Preprocessing & Feature Engineering
#========================================================================
def preprocess(df):
    df = df.copy()
    cols_to_drop = ['PassengerId', 'Ticket', 'Cabin', 'Name']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    df['Age']      = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['family_size']    = df['SibSp'] + df['Parch'] + 1
    df['is_alone']       = (df['family_size'] == 1).astype(int)
    df['age_bin']        = pd.cut(df['Age'], bins=[0, 12, 60, 100],
                                  labels=['child', 'adult', 'senior'])
    df['fare_per_person'] = df['Fare'] / df['family_size']
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
    df = pd.get_dummies(df, columns=['Embarked', 'age_bin'], drop_first=True)
    return df

train_data = preprocess(train)
test_data  = preprocess(test)

# Align columns — test may be missing some dummy columns
train_data, test_data = train_data.align(test_data, join='left', axis=1, fill_value=0)

X_fe = train_data.drop('Survived', axis=1).fillna(train_data.median(numeric_only=True))
y_fe = train_data['Survived']
print(X_fe.dtypes)
X_train_fe, X_test_fe, y_train_fe, y_test_fe = train_test_split(X_fe, y_fe, test_size=0.2, random_state=42)

# Scale -- fit only on train
scaler_fe = StandardScaler()
X_train_fe_scaled = scaler_fe.fit_transform(X_train_fe)
X_test_fe_scaled  = scaler_fe.transform(X_test_fe)
print('Feature set:', list(X_train_fe.columns))

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
 # SECTION 4 -- Evaluation (CV on train only — test has no labels)
#========================================================================
# test.csv has no 'Survived' column so we cannot compute test metrics directly.
# Cross-validation on the training set is the performance estimate.

cv_results = {}
for name, model in models.items():
    if name in ['Decision Tree', 'Random Forest']:
        scores = cross_val_score(model, X_train_fe, y_train_fe, cv=5, scoring='f1')
    else:
        scores = cross_val_score(model, X_train_fe_scaled, y_train_fe, cv=5, scoring='f1')
    cv_results[name] = scores
    print(f'{name}: CV F1 = {scores.mean():.3f} +/- {scores.std():.3f}')

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
                       param_grid_dt, cv=5, scoring='f1')
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
                        param_grid_knn, cv=5, scoring='f1')
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
                       param_grid_lr, cv=5, scoring='f1')
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
                       param_grid_rf, cv=5, scoring='f1')
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
                        param_grid_svm, cv=5, scoring='f1')
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
# Select best model based on the robust GridSearch CV Score instead of the tiny test split
cv_best_scores = {
    'Tuned Decision Tree':       grid_dt.best_score_,
    'Tuned k-NN':                grid_knn.best_score_,
    'Tuned Logistic Regression': grid_lr.best_score_,
    'Tuned Random Forest':       grid_rf.best_score_,
    'Tuned SVM':                 grid_svm.best_score_,
}

best_tuned_name = max(cv_best_scores, key=cv_best_scores.get)
best_cv_score = cv_best_scores[best_tuned_name]

print(f'\nSelected model: {best_tuned_name}  (Robust CV F1 = {best_cv_score:.3f})')

# Predict on actual test.csv using the best tuned model
X_final = test_data.drop('Survived', axis=1, errors='ignore').fillna(X_train_fe.median())
X_final_scaled = scaler_fe.transform(X_final)

best_model, _ = tuned_models[best_tuned_name]
if best_tuned_name in ['Tuned Decision Tree', 'Tuned Random Forest']:
    final_pred = best_model.predict(X_final)
else:
    final_pred = best_model.predict(X_final_scaled)

output = pd.DataFrame({'predicted': final_pred})
output.to_csv('predictions.csv', index=False)
print('Predictions saved to predictions.csv')

