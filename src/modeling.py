from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils_io import load_processed
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)


def evaluate(name, y_true, y_pred, y_prob=None):
    """
    Parameters:
        - name   : Model name
        - y_true : Actual y values
        - y_pred : Predicted y using model
        - y_prob : Prob of y

    Helper method for evaluating model using Accuracy, Precision, Recall, F1 score, AUROC
    """
    print(f"\n===== {name} =====")
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy : {acc:.4f}")

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    print(f"Precision: {prec:.4f}\nRecall: {rec:.4f}\nF1 Score: {f1:.4f}")

    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

    if y_prob is not None:
        # y_prob should be probability for the positive class (shape (n_samples,))
        auroc = roc_auc_score(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        print(f"AUROC    : {auroc:.4f}")
        print(f"PR AUC   : {ap:.4f}")


def plot_curves(
    y_test,
    y_prob_lr,
    y_prob_rf,
    y_pred_lr=None,
    y_pred_rf=None,
    save_dir: Path | str = "reports/figures",
    show: bool = True,
):
    """
    Parameters:
        - y_test     : array-like of true labels (0/1)
        - y_prob_lr  : LR predicted probs for positive class (shape (n,))
        - y_prob_rf  : RF predicted probs for positive class (shape (n,))
        - y_pred_lr  : optional LR hard predictions (0/1) for confusion matrix
        - y_pred_rf  : optional RF hard predictions (0/1) for confusion matrix
        - save_dir   : folder to save figures (created if needed)
        - show       : whether to display figures interactively

    Produces:
        1) ROC and Precision–Recall curves (one figure, two subplots)
        2) Confusion matrices for LR and RF (one figure, two subplots)
        3) Calibration curve (reliability diagram) comparing LR and RF
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ROC + PR 
    plt.figure(figsize=(12, 5))

    # ROC
    plt.subplot(1, 2, 1)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    plt.plot(fpr_lr, tpr_lr, label="LogReg")
    plt.plot(fpr_rf, tpr_rf, label="RandomForest")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    # PR
    plt.subplot(1, 2, 2)
    prec_lr, rec_lr, _ = precision_recall_curve(y_test, y_prob_lr)
    prec_rf, rec_rf, _ = precision_recall_curve(y_test, y_prob_rf)
    plt.plot(rec_lr, prec_lr, label="LogReg")
    plt.plot(rec_rf, prec_rf, label="RandomForest")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_dir / "roc_pr_curves.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    # Confusion matrices 
    if (y_pred_lr is not None) and (y_pred_rf is not None):
        cm_lr = confusion_matrix(y_test, y_pred_lr)
        cm_rf = confusion_matrix(y_test, y_pred_rf)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, cm, title in zip(
            axes,
            (cm_lr, cm_rf),
            ("Confusion Matrix – LogReg", "Confusion Matrix – RandomForest"),
        ):
            im = ax.imshow(cm, cmap="Blues")
            ax.set_title(title)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, int(v), ha="center", va="center", fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(save_dir / "confusion_matrices.png", dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()

    # Calibration (reliability) curve 
    # Use 10 bins for smoother lines
    prob_true_lr, prob_pred_lr = calibration_curve(y_test, y_prob_lr, n_bins=10, strategy="uniform")
    prob_true_rf, prob_pred_rf = calibration_curve(y_test, y_prob_rf, n_bins=10, strategy="uniform")

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly calibrated")
    plt.plot(prob_pred_lr, prob_true_lr, marker="o", label="LogReg")
    plt.plot(prob_pred_rf, prob_true_rf, marker="o", label="RandomForest")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Empirical Positive Rate")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "calibration_curve.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()




def plot_feature_importances(importances, top_n=20, save_path=None):
    """
    Parameters:
        - importances : pd.Series of RF importances, sorted
        - top_n       : Number of features to plot
        - save_path   : Optional path to save the figure
    """
    top = importances.head(top_n)
    plt.figure(figsize=(8,6))
    top[::-1].plot(kind="barh")  
    plt.title(f"Top {top_n} RandomForest Feature Importances")
    plt.xlabel("Importance")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Feature importance plot saved to {save_path}")
    else:
        plt.show()


def main():
    # Load Dataset
    df = load_processed()

    # Split features/target
    y = df["is_accident"].astype(int)
    X = df.drop(columns=["is_accident"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit Logistic Regression 
    scaler = StandardScaler(with_mean=False) 
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    lr = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        class_weight=None # we have a 1:1 ratio for pos:neg 
    )
    lr.fit(X_train_sc, y_train)
    lr_pred = lr.predict(X_test_sc)
    lr_prob = lr.predict_proba(X_test_sc)[:, 1]
    evaluate("Logistic Regression", y_test, lr_pred, lr_prob)

    # Fit Random Forest 
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    evaluate("Random Forest", y_test, rf_pred, rf_prob)

    # Get feature importance in descending order
    try:
        importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        print("\nTop 20 RF feature importances:")
        print(importances.head(20).round(4))
    except Exception as e:
        print("Could not compute feature importances:", e)

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    # print out top 20 features in descending importance
    plot_feature_importances(importances, top_n=20, save_path="reports/figures/feature_importances.png")

    # plot curves (ROC and PR)
    plot_curves(y_test, lr_prob, rf_prob, y_pred_lr=lr_pred, y_pred_rf=rf_pred, save_dir="reports/figures", show=True)

if __name__ == "__main__":
    main()
