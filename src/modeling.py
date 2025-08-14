from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from src.utils_io import load_processed

def main():
    ###################################################
    #################### L0AD DATA ####################
    ###################################################

    df = load_processed()

    # Explanatory and Response Variable
    X = df.drop(columns=["GRAVITE"])
    y = df["GRAVITE"]

    # Make sure GRAVITE is numeric
    if y.dtype == "object" or y.dtype.name == "category":
        y = y.astype("category").cat.codes
    
    # Split data in Test and Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("\nX_train shape:", X_train.shape)
    print("\nX_test shape:", X_test.shape)

    # Scale data before fitting model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    #############################################################
    #################### LOGISTIC REGRESSION ####################
    #############################################################
    lr_model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)

    # balanced model
    lr_model_bal = LogisticRegression(solver="lbfgs", class_weight="balanced",max_iter=1000, random_state=42)
    lr_model_bal.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = lr_model.predict(X_test_scaled)
    y_pred_bal = lr_model_bal.predict(X_test_scaled)
    
    # Multiclass AUROC (weighted One-vs-Rest)


    #############################################################
    ####################### RANDOM FOREST #######################
    #############################################################

    rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
    )

    rf_model.fit(X_train, y_train)

    # Predictions
    y_pred_rf = rf_model.predict(X_test)

    #######################################################
    ####################### METRICS #######################
    #######################################################

    # Logistic Regression Metrics
    print("\nClassification Report for Unbalanced Model:\n", classification_report(y_test, y_pred, zero_division=0))
    print("\nConfusion Matrix for Unbalanced model:\n", confusion_matrix(y_test, y_pred))

    print("\nClassification Report for Balanced Model:\n", classification_report(y_test, y_pred_bal, zero_division=0))
    print("\nConfusion Matrix for Balanced model:\n", confusion_matrix(y_test, y_pred_bal))

    # Random Forest Metrics
    print("\nClassification Report (RandomForest, balanced):\n", classification_report(y_test, y_pred_rf, zero_division=0))
    print("\nConfusion Matrix (RandomForest, balanced):\n", confusion_matrix(y_test, y_pred_rf))

    # Extract feature importances
    importances = rf_model.feature_importances_
    feature_names = X_train.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

    # Rank features by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    print(feature_importance_df)

    # Select top N features (example selecting top 10 features)
    top_features = feature_importance_df['Feature'][:50].values
    X_train_selected = X_train[top_features]
    X_test_selected = X_test[top_features]
    print(X_train_selected)
    print(X_test_selected)

    ####################################################################
    ####################### TUNE HYPERPARAMETERS #######################
    ####################################################################
    # --- Randomized hyperparameter search for RandomForest ---
    param_dist = {
        "n_estimators": [200, 300, 500, 700],
        "max_depth": [None, 10, 20, 30, 40],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
        # “balanced_subsample” reweights per bootstrap sample; can help
        "class_weight": ["balanced", "balanced_subsample"]
    }

    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

    rf_search = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=param_dist,
        n_iter=5,             # quick first pass
        cv=3,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1,
        verbose=2
    )

    print("\nTuning RandomForest... (this can take a few minutes)")
    rf_search.fit(X_train, y_train)

    print("\nBest RF params:", rf_search.best_params_)
    print("Best CV score:", round(rf_search.best_score_, 4))

    # Retrain best model on full training set
    best_rf = rf_search.best_estimator_
    best_rf.fit(X_train, y_train)

    # Evaluate on test
    y_pred_rf_tuned = best_rf.predict(X_test)
    print("\nClassification Report (RandomForest TUNED):\n",
        classification_report(y_test, y_pred_rf_tuned, zero_division=0))
    print("\nConfusion Matrix (RandomForest TUNED):\n",
        confusion_matrix(y_test, y_pred_rf_tuned))









    # # Train the Random Forest model with selected features
    # rf_selected = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1)
    # rf_selected.fit(X_train_selected, y_train)

    # # Evaluate the model
    # accuracy_after = rf_selected.score(X_test_selected, y_test)
    # print(f'Accuracy after feature selection: {accuracy_after:.2f}')

if __name__ == "__main__":
    main()