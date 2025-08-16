from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from imblearn.over_sampling import SMOTE


from src.utils_io import load_processed

def main():
    ###################################################
    #################### L0AD DATA ####################
    ###################################################

    df = load_processed()

    # Explanatory and Response Variable
    X = df.drop(columns=["GRAVITE"])
    y = df["GRAVITE"]
    
    # Split data in Test and Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    oversample = SMOTE()
    X_train_SMOTE, y_train_SMOTE = oversample.fit_resample(X_train, y_train)
    counter = Counter(y_train_SMOTE)

    print("\nX_train shape:", X_train.shape)
    print("\nX_test shape:", X_test.shape)

    print("\nCheck for spread of data points across classes before SMOTE: ", y_train.value_counts())
    print("\nCheck for spread of data points across classes after SMOTE: ", y_train_SMOTE.value_counts())

    # Scale data before fitting model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled_SMOTE = scaler.fit_transform(X_train_SMOTE)

    #############################################################
    #################### LOGISTIC REGRESSION ####################
    #############################################################
    lr_model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)

    lr_model_SMOTE = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
    lr_model_SMOTE.fit(X_train_scaled_SMOTE, y_train_SMOTE)


    # Predictions
    y_pred = lr_model.predict(X_test_scaled)
    y_pred_SMOTE = lr_model_SMOTE.predict(X_test_scaled)
    
    # Multiclass AUROC (weighted One-vs-Rest)


    # #############################################################
    # ####################### RANDOM FOREST #######################
    # #############################################################

    # rf_model = RandomForestClassifier(
    # n_estimators=200,
    # random_state=42,
    # class_weight="balanced",
    # n_jobs=-1
    # )

    # rf_model.fit(X_train, y_train)

    # # Predictions
    # y_pred_rf = rf_model.predict(X_test)

    #######################################################
    ####################### METRICS #######################
    #######################################################

    # Logistic Regression Metrics
    print("\nClassification Report for Logistic Regression Model:\n", classification_report(y_test, y_pred, zero_division=0))
    print("\nConfusion Matrix for Logistic Regression Model:\n", confusion_matrix(y_test, y_pred))

    print("\nClassification Report for Logistic Regression Model (SMOTE):\n", classification_report(y_test, y_pred_SMOTE, zero_division=0))
    print("\nConfusion Matrix for Logistic Regression Model (SMOTE):\n", confusion_matrix(y_test, y_pred_SMOTE))

    # # Random Forest Metrics
    # print("\nClassification Report (RandomForest, balanced):\n", classification_report(y_test, y_pred_rf, zero_division=0))
    # print("\nConfusion Matrix (RandomForest, balanced):\n", confusion_matrix(y_test, y_pred_rf))

    ####################################################################
    ####################### TUNE HYPERPARAMETERS #######################
    ####################################################################
    # --- Randomized hyperparameter search for RandomForest ---
    # param_dist = {
    #     "n_estimators": [200, 300, 500, 700],
    #     "max_depth": [None, 10, 20, 30, 40],
    #     "min_samples_split": [2, 5, 10],
    #     "min_samples_leaf": [1, 2, 4],
    #     "max_features": ["sqrt", "log2", None],
    #     # “balanced_subsample” reweights per bootstrap sample; can help
    #     "class_weight": ["balanced", "balanced_subsample"]
    # }

    # rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

    # rf_search = RandomizedSearchCV(
    #     estimator=rf_base,
    #     param_distributions=param_dist,
    #     n_iter=5,             # quick first pass
    #     cv=3,
    #     scoring="accuracy",
    #     random_state=42,
    #     n_jobs=-1,
    #     verbose=2
    # )

    # print("\nTuning RandomForest... (this can take a few minutes)")
    # rf_search.fit(X_train, y_train)

    # print("\nBest RF params:", rf_search.best_params_)
    # print("Best CV score:", round(rf_search.best_score_, 4))

    # # Retrain best model on full training set
    # best_rf = rf_search.best_estimator_
    # best_rf.fit(X_train, y_train)

    # # Evaluate on test
    # y_pred_rf_tuned = best_rf.predict(X_test)
    # print("\nClassification Report (RandomForest TUNED):\n",
    #     classification_report(y_test, y_pred_rf_tuned, zero_division=0))
    # print("\nConfusion Matrix (RandomForest TUNED):\n",
    #     confusion_matrix(y_test, y_pred_rf_tuned))







if __name__ == "__main__":
    main()