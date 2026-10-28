import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import PrecisionRecallDisplay

from .common import ROOT, CLASSES, FEATURES, TARGET

FIG = ROOT / "results/figures"
sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")


def save(name):
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png", dpi=170, bbox_inches="tight")
    plt.close()


def exploration(df):
    counts = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(CLASSES, counts, color=["#527c96", "#d18a32", "#ad5864"])
    ax.set(title="BRFSS 2015: three-class distribution", ylabel="Survey responses")
    for i, count in enumerate(counts):
        ax.text(i, count, f"{count:,}\n({count/len(df):.2%})", ha="center", va="bottom")
    ax.set_ylim(0, counts.max() * 1.22)
    save("class_distribution")
    columns = ["BMI", "Age", "GenHlth", "PhysHlth"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, col in zip(axes.flat, columns):
        sns.boxplot(data=df, x=TARGET, y=col, showfliers=False, ax=ax, color="#aac9d6")
        ax.set_xticks([0, 1, 2], CLASSES)
        ax.set_xlabel("")
        if col == "Age":
            ax.set_ylabel("Age category (1–13)")
    fig.suptitle("Selected indicator distributions by reported status", y=1.02)
    save("indicator_distributions")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(df[FEATURES].corr(), cmap="vlag", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Feature correlations (ordinal codes treated numerically)")
    save("feature_correlations")


def confusion(cm, name, split):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES,
                yticklabels=CLASSES, cbar=False, ax=axes[0])
    norm = np.divide(cm, cm.sum(axis=1, keepdims=True), out=np.zeros_like(cm, dtype=float), where=cm.sum(axis=1, keepdims=True) != 0)
    sns.heatmap(norm, annot=True, fmt=".1%", vmin=0, vmax=1, cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, cbar=False, ax=axes[1])
    for ax, title in zip(axes, ["Counts", "Within each true class"]):
        ax.set(xlabel="Predicted label", ylabel="True label", title=title)
        ax.tick_params(axis="y", rotation=0)
    fig.suptitle(f"{name} — {split}")
    save(f"{name}_{split}_confusion")


def history_plot(history, name):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.epoch, history.train_objective, label="Train objective")
    axes[0].plot(history.epoch, history.validation_ce, label="Tuning ordinary CE")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Training progress")
    axes[0].legend()
    axes[1].plot(history.epoch, history.validation_macro_f1)
    axes[1].set(xlabel="Epoch", ylabel="Macro F1", title="Checkpoint selection on tuning split")
    fig.suptitle(name)
    save(f"{name}_loss")


def coefficients(model, scaler, columns):
    # Inputs were standardized; binary feature coefficients are per SD too.
    frame = pd.DataFrame(model.coef_.T, index=columns, columns=CLASSES)
    frame.to_csv(ROOT / "results/logistic_coefficients.csv", index_label="feature")
    top = frame["Diabetes"].abs().nlargest(12).index
    values = frame.loc[top, "Diabetes"].sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(values.index, values, color=np.where(values < 0, "#ad5864", "#527c96"))
    ax.set(title="Logistic regression: diabetes-class coefficients", xlabel="Coefficient per input standard deviation")
    save("logistic_coefficients")


def comparison(table):
    columns = ["macro_f1", "balanced_accuracy", "accuracy", "prediabetes_recall"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for col, ax in zip(columns, axes.flat):
        ax.barh(table.model, table[col], color="#527c96")
        ax.set(xlim=(0, 1), xlabel=col.replace("_", " ").title())
        for i, v in enumerate(table[col]):
            ax.text(min(v + .01, .92), i, f"{v:.3f}", va="center", fontsize=9)
    fig.suptitle("Frozen models — held-out test results")
    save("test_comparison")


def precision_recall(y, proba, name):
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, label in enumerate(CLASSES):
        PrecisionRecallDisplay.from_predictions(np.asarray(y) == i, proba[:, i], name=label, ax=ax)
    ax.set(title=f"{name}: one-vs-rest precision–recall", xlim=(0, 1), ylim=(0, 1))
    save("champion_precision_recall")
