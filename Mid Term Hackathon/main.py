import argparse
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score
import warnings

def main():
    # Ignore warnings
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser()
    parser.add_argument('train_path', type=str)
    parser.add_argument('test_path', type=str)
    parser.add_argument('--target', type=str, default='target')
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)

    if 'ID' in train_df.columns:
        train_df = train_df.drop(columns=['ID'])

    X_full = train_df.drop(columns=[args.target])
    y_full = train_df[args.target]

    X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)

    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_features)
        ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10]
    }

    # Use GridSearchCV for exhaustive checking
    search = GridSearchCV(pipeline, param_grid=param_grid, 
                          cv=5, scoring='accuracy', n_jobs=-1)
    
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    print(f"Best CV Accuracy: {search.best_score_:.4f}")
    print(f"Best Parameters: {search.best_params_}")

    val_predictions = best_model.predict(X_val)
    print(f"Validation Accuracy: {accuracy_score(y_val, val_predictions):.4f}")
    print(f"Validation F1 Score: {f1_score(y_val, val_predictions):.4f}")

    # Retrain best pipeline on full data
    best_model.fit(X_full, y_full)
    
    test_features = test_df.copy()
    if 'ID' in test_features.columns:
        test_features = test_features.drop(columns=['ID'])
        
    final_predictions = best_model.predict(test_features)

    output_df = pd.DataFrame({'Prediction': final_predictions})
    if 'ID' in test_df.columns:
        output_df.insert(0, 'ID', test_df['ID'])
        
    output_df.to_csv('final_predictions.csv', index=False)

if __name__ == "__main__":
    main()
