import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier # Change to RandomForestRegressor for regression
from sklearn.metrics import accuracy_score, classification_report # Change to mean_squared_error for regression
import joblib
import time

def main():
    # 1. PROBLEM FRAMING & INPUTS (Command Line Arguments)
    parser = argparse.ArgumentParser(description="Auto ML Pipeline")
    parser.add_argument('train_path', type=str, help="Path to training CSV")
    parser.add_argument('test_path', type=str, help="Path to testing CSV")
    # Assume the target column is named 'target'. Change this during the exam if needed!
    parser.add_argument('--target', type=str, default='target', help="Name of the target column")
    args = parser.parse_args()

    print("Loading data...")
    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)

    # Separate features (X) and labels (y)
    X = train_df.drop(columns=[args.target])
    y = train_df[args.target]

    # Train-Test Split (for internal evaluation before final prediction)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 2. AUTOMATED DATA CLEANING & FEATURE ENGINEERING
    # Auto-detect numerical and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Preprocessing for numerical data: Impute missing with median, then scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical data: Impute missing with 'missing_value', then OHE
    # handle_unknown='ignore' PREVENTS the model from crashing if test set has new categories
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 3. MODEL BUILDING (The Pipeline)
    # Using Random Forest as the robust baseline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # 4. HYPERPARAMETER TUNING
    # Using RandomizedSearchCV because GridSearchCV is too slow for a 2-hour exam
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5, 10]
    }

    print("Starting Training and Tuning (this may take a minute)...")
    start_time = time.time()
    
    search = RandomizedSearchCV(pipeline, param_distributions=param_grid, 
                                n_iter=5, cv=3, scoring='accuracy', n_jobs=-1, random_state=42)
    search.fit(X_train, y_train)
    
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")
    print(f"Best Parameters: {search.best_params_}")

    # 5. MODEL EVALUATION (Internal Monitoring)
    best_model = search.best_estimator_
    val_predictions = best_model.predict(X_val)
    
    print("\n--- Validation Performance ---")
    print(f"Accuracy: {accuracy_score(y_val, val_predictions):.4f}")
    print(classification_report(y_val, val_predictions))

    # 6. FINAL INFERENCE ON TEST DATA
    print("Predicting on unseen test data...")
    # The pipeline automatically handles missing values and scaling for the test set!
    final_predictions = best_model.predict(test_df)

    # Save predictions alongside whatever ID column exists in the test set
    # Assuming test set has an 'ID' column. If not, just save the raw predictions.
    output_df = pd.DataFrame({'Prediction': final_predictions})
    if 'ID' in test_df.columns:
        output_df.insert(0, 'ID', test_df['ID'])
        
    output_df.to_csv('final_predictions.csv', index=False)
    print("Predictions saved to 'final_predictions.csv'")

    # 7. MODEL DEPLOYMENT (Saving the artifact)
    joblib.dump(best_model, 'deployed_model.pkl')
    print("Model pipeline saved as 'deployed_model.pkl' for future deployment.")

if __name__ == "__main__":
    main()
