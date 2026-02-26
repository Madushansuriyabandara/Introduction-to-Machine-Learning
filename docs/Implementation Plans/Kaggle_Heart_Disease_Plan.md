# Kaggle Heart Disease Prediction (S6E2) - Improvement Plan

This plan documents the strategies to improve the evaluation score (ROC AUC) for the synthetic heart disease prediction dataset.

## Current Setup Analysis
1. **Target Metric**: The Kaggle competition "Playground Series Season 6 Episode 2" uses the **ROC AUC** (Receiver Operating Characteristic Area Under the Curve) score. This is evident from your model using `.predict_proba()` and optimizing for probabilities.
2. **Current Models**: You are using a well-structured Soft-Voting Ensemble of `HistGradientBoostingClassifier` and `RandomForestClassifier`.
3. **Data**: We have 630,000 rows, 14 features, and 0 missing values.
4. **Current Preprocessing**: You apply `StandardScaler` to all features. However, several features (like `Chest pain type`, `EKG results`, `Thallium`, etc.) are purely categorical and standardizing them as continuous integers can confuse the model.

## User Review Required
> [!IMPORTANT]
> The prompt guidelines specify that we should start with smaller modifications but chat before taking larger steps.
> Please review the two approaches below and let me know if you would like me to proceed with the **Smaller Modifications (Phase 1)** first, or if you want to jump straight to the **Larger Modifications (Phase 2)**.

## Proposed Changes

### Phase 1: Smaller Modifications (Recommended First Step)
These are straightforward tweaks to your existing code that typically yield a solid boost in performance:

1. **Proper Feature Encoding**: 
   - Separate the 14 columns into Continuous and Categorical features.
   - Apply `StandardScaler` **only** to continuous features (Age, BP, Cholesterol, Max HR, ST depression).
   - Apply `OneHotEncoder` to categorical features (Sex, Chest pain type, FBS, EKG results, Exercise angina, Slope of ST, Number of vessels, Thallium).
2. **Local Validation Strategy**:
   - Rather than just fitting on the entire dataset, implement a **Stratified 5-Fold Cross-Validation** loop. This gives us a reliable local ROC AUC score so we know whether our changes are actually improving the model without having to submit to Kaggle every time.
3. **Hyperparameter Tweaks**:
   - Tweak `max_iter`, `learning_rate`, and tree depth for your current `HistGradientBoostingClassifier` and `RandomForestClassifier` to reduce overfitting.

### Phase 2: Larger Modifications (Optional)
If the Phase 1 changes don't push the score past 0.954, we can employ Kaggle-winning strategies for tabular data:

1. **Integrating Top-Tier Algorithms**:
   - Replace or append to your ensemble using **XGBoost** (`XGBClassifier`) and **LightGBM** (`LGBMClassifier`). These often outperform raw Scikit-Learn implementations.
2. **Feature Engineering**:
   - Create Interaction Features: e.g., combining `Age` and `Max HR`, or `BP` and `Cholesterol` ratios.
3. **Including Original Data (Advanced)**:
   - The dataset is synthetic. According to Kaggle discussions, adding the original UCI Heart Disease dataset to the training set often reduces the "noise" and corrects artifacts.

## Verification Plan

### Automated Tests
- We will execute the Python script locally. The Stratified K-Fold CV section will output the average OOF (Out-Of-Fold) ROC AUC score. 
- A successful change will be indicated by the OOF score increasing above the baseline we observe initially.

### Manual Verification
- Once the script finishes, it will generate a new `submission.csv`. You will submit this file to Kaggle to verify the public LB score hits the `>0.954` target.
