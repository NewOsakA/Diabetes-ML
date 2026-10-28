"""Build beginner-facing scientific illustrations from the committed evidence."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/wiki/assets"
ASSETS.mkdir(parents=True, exist_ok=True)
BLUE, AMBER, RED, PALE = "#326a93", "#d28c2d", "#ab5262", "#dce7ef"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
                     "axes.spines.top": False, "axes.spines.right": False})


def save(name):
    plt.tight_layout()
    plt.savefig(ASSETS / name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()


def flow(name, title, nodes, edges, size=(10, 7), loop=False):
    """Static flow diagrams remain readable without third-party iframe rendering."""
    fig, ax = plt.subplots(figsize=size)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.set_title(title, fontsize=18, pad=20)
    for a, b in edges:
        xa, ya, _, wa, ha = nodes[a]
        xb, yb, _, wb, hb = nodes[b]
        dx, dy = xb-xa, yb-ya
        ta = min(wa/(2*abs(dx)) if dx else float("inf"), ha/(2*abs(dy)) if dy else float("inf"))
        tb = min(wb/(2*abs(dx)) if dx else float("inf"), hb/(2*abs(dy)) if dy else float("inf"))
        ax.annotate("", xy=(xb-dx*tb, yb-dy*tb), xytext=(xa+dx*ta, ya+dy*ta),
                    arrowprops={"arrowstyle":"->", "color":"#52606d", "lw":2}, zorder=1)
    for x, y, label, width, height in nodes.values():
        box = FancyBboxPatch((x-width/2,y-height/2),width,height,boxstyle="round,pad=.012",
                             facecolor=PALE,edgecolor=BLUE,linewidth=1.5,zorder=2)
        ax.add_patch(box)
        ax.text(x,y,label,ha="center",va="center",fontsize=13,color="#163b58",zorder=3)
    if loop:
        ax.plot([.84,.94,.94],[.16,.16,.70],color=AMBER,lw=2)
        ax.annotate("",xy=(.84,.70),xytext=(.94,.70),arrowprops={"arrowstyle":"->","color":AMBER,"lw":2})
        ax.text(.965,.43,"Next batch",rotation=90,ha="center",va="center",color="#8e5c17",fontsize=11)
    save(name)


def main():
    # Rounded composition is explicitly labeled; exact totals are provided in the wiki.
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [BLUE]*84 + [AMBER]*2 + [RED]*14
    for i, color in enumerate(colors):
        ax.scatter(i % 20, 4-i//20, s=210, c=color, marker="o")
    ax.set_title("Imagine 100 survey responses", loc="left", fontsize=20, pad=22)
    ax.set(xlim=(-1, 20), ylim=(-1.3, 5))
    ax.axis("off")
    ax.legend(handles=[Patch(color=BLUE, label="84 no diabetes"), Patch(color=AMBER, label="2 prediabetes"),
                       Patch(color=RED, label="14 diabetes")], loc="lower left", ncol=3, frameon=False)
    fig.text(.05, .01, "Rounded illustration of the dataset balance; each dot represents about one response in 100.", fontsize=11, color="#52606d")
    save("class-balance.png")

    table = pd.read_csv(ROOT / "results/test_results.csv").set_index("model")
    base, improved = table.loc["mlp_baseline"], table.loc["champion"]
    metrics = ["accuracy", "diabetes_recall", "prediabetes_recall"]
    titles = ["All responses correct", "Actual diabetes cases found", "Actual prediabetes cases found"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, metric, title in zip(axes, metrics, titles):
        values = [base[metric]*100, improved[metric]*100]
        bars = ax.bar(["Baseline", "Weighted MLP"], values, color=[PALE, BLUE], width=.6)
        ax.set(ylim=(0, 100), title=title, ylabel="Percent")
        ax.tick_params(axis="x", labelsize=10)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, value+2, f"{value:.1f}%", ha="center", fontsize=14)
    fig.suptitle("Experiment 1: better detection comes with more errors elsewhere", fontsize=18, y=1.05)
    save("experiment-1-tradeoff.png")

    cm = pd.read_csv(ROOT / "results/champion_test_confusion.csv").to_numpy()
    tp, actual, predicted = int(cm[2, 2]), int(cm[2].sum()), int(cm[:, 2].sum())
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.6))
    for ax, denominator, label, rest, color in [
        (axes[0], actual, "RECALL: start with actual diabetes cases", "Missed", AMBER),
        (axes[1], predicted, "PRECISION: start with diabetes predictions", "Other actual statuses", RED),
    ]:
        correct = tp/denominator
        ax.barh(0, correct, color=BLUE, height=.45)
        ax.barh(0, 1-correct, left=correct, color=color, height=.45)
        ax.text(correct/2, 0, f"Correct: {tp:,}", color="white", ha="center", va="center", weight="bold")
        ax.text(correct+(1-correct)/2, 0, f"{rest}: {denominator-tp:,}", color="white", ha="center", va="center", fontsize=11)
        ax.set(xlim=(0, 1), ylim=(-.6, .6), title=f"{label}\n{tp:,} / {denominator:,} = {correct:.1%}", yticks=[])
        ax.set_xticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
    fig.suptitle("Same correct predictions. Different questions and denominators.", fontsize=18, y=1.04)
    save("precision-versus-recall.png")

    fig, ax = plt.subplots(figsize=(11, 3.7))
    segments = [(.8, "Train\n202,944", BLUE), (.05, "Tune\n12,684", AMBER),
                (.05, "Select\n12,684", "#75947a"), (.1, "Test\n25,368", RED)]
    left = 0.
    for width, label, color in segments:
        ax.barh(0, width, left=left, height=.45, color=color)
        center = left+width/2
        if width >= .1:
            ax.text(center, 0, label, ha="center", va="center", color="white", weight="bold", fontsize=12)
        else:
            ytext = .65 if left < .85 else -.65
            ax.annotate(label, xy=(center, .23 if ytext>0 else -.23), xytext=(center, ytext),
                        ha="center", va="center", fontsize=11, arrowprops={"arrowstyle":"-", "color":color})
        left += width
    ax.set(xlim=(0, 1), ylim=(-1, 1), title="Experiment 1: different data portions have different jobs")
    ax.axis("off")
    fig.text(.12, .02, "Train = learn weights    Tune = adjust settings    Select = choose model    Test = final evaluation", fontsize=11)
    save("data-partitions.png")

    outer = pd.read_csv(ROOT / "experiments/v2/outer_results.csv")
    methods = ["fixed_weighted_mlp", "selected_mlp", "selected_catboost", "selected_overall"]
    names = ["Fixed weighted MLP", "Tuned MLP procedure", "Tuned CatBoost procedure", "Selected overall procedure"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (method, label) in enumerate(zip(methods, names)):
        values = outer.loc[outer.method.eq(method), "macro_f1"]
        ax.barh(i, values.mean(), color=BLUE if method=="selected_overall" else PALE, height=.6)
        ax.errorbar(values.mean(), i, xerr=values.std(), fmt="o", color="#163b58", capsize=4)
        ax.text(values.mean()+.008, i, f"{values.mean():.4f}", va="center", fontsize=13)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set(xlim=(0, .55), xlabel="Macro F1 (higher is better)", title="Experiment 2: the extra gains are small")
    fig.text(.08, -.02, "Mean of three outer folds. Error bars show fold standard deviation, not a confidence interval.", fontsize=11)
    save("experiment-2-comparison.png")

    # A simple numeric learning sketch supports readers who do not render Mermaid.
    fig, ax = plt.subplots(figsize=(10, 3.7))
    blocks = [(.03, "21 inputs", "Survey indicators"), (.29, "64 units", "Hidden layer 1"),
              (.55, "32 units", "Hidden layer 2"), (.81, "3 scores", "One per class")]
    for i, (x, title, subtitle) in enumerate(blocks):
        ax.text(x+.08, .62, title, transform=ax.transAxes, ha="center", va="center", fontsize=18,
                color="white", bbox={"boxstyle":"round,pad=.65", "facecolor":BLUE, "edgecolor":"none"})
        ax.text(x+.08, .25, subtitle, transform=ax.transAxes, ha="center", fontsize=11, color="#52606d")
        if i < 3:
            ax.annotate("", xy=(x+.25,.62), xytext=(x+.18,.62), xycoords="axes fraction",
                        arrowprops={"arrowstyle":"->", "color":"#52606d", "lw":2})
    ax.set_title("Experiment 1 weighted MLP: information flows through layers", fontsize=18, pad=20)
    ax.axis("off")
    save("neural-network.png")

    flow("two-experiments.png", "Two experiments answer different evaluation questions", {
        "data": (.5,.87,"Health indicator dataset",.52,.12),
        "one": (.25,.60,"Experiment 1\nFixed train / validation / test split",.43,.16),
        "two": (.75,.60,"Experiment 2\nDevelopment-only nested CV",.43,.16),
        "q1": (.25,.25,"How do the baseline and improved\nmodels compare on held-out rows?",.43,.23),
        "q2": (.75,.25,"How stable is the broader\nmodel-selection procedure?",.43,.23),
    }, [("data","one"),("data","two"),("one","q1"),("two","q2")])
    flow("learning-loop.png", "How supervised neural learning works", {
        "a": (.5,.88,"Read 21 health indicators",.65,.12),
        "b": (.5,.70,"Produce three class scores",.65,.12),
        "c": (.5,.52,"Compare with the known survey label",.65,.12),
        "d": (.5,.34,"Calculate the prediction loss",.65,.12),
        "e": (.5,.16,"Optimizer adjusts network parameters",.65,.12),
    }, [("a","b"),("b","c"),("c","d"),("d","e")],size=(9,7),loop=True)
    flow("class-weighting.png", "Class weighting changes the training loss", {
        "a": (.12,.5,"Same training\nexamples",.20,.38),
        "b": (.37,.5,"Assign class\nweights to errors",.20,.38),
        "c": (.62,.5,"Rare-class errors\ncontribute more",.20,.38),
        "d": (.87,.5,"Evaluate on\nseparate rows",.20,.38),
    },[("a","b"),("b","c"),("c","d")],size=(12,3.5))
    labels = ["Split development data into three outer folds", "Hold one outer fold aside for evaluation",
              "Use two inner folds on the remaining data\nto choose model settings and prediction policy",
              "Refit within the outer training data\nusing separate fit / stopping / calibration / decision roles",
              "Evaluate once on the held-aside outer fold",
              "Repeat until all three outer folds are evaluated"]
    nodes={str(i):(.5,.90-i*.16,label,.88,.105) for i,label in enumerate(labels)}
    flow("nested-evaluation.png", "Nested cross-validation: choose inside, evaluate outside",nodes,
         [(str(i),str(i+1)) for i in range(5)],size=(10,9))
    flow("evidence-map.png", "Keep model selection separate from evaluation", {
        "a": (.5,.88,"inner_results.csv",.45,.13),
        "b": (.5,.66,"Choose settings using inner scores",.65,.13),
        "c": (.24,.41,"outer_results.csv",.42,.15),
        "d": (.24,.14,"Evaluate the selection\nprocedure on separate folds",.42,.17),
        "e": (.76,.41,"Final development search",.42,.15),
        "f": (.76,.14,"Save fitted models\nand decision settings",.42,.17),
    },[("a","b"),("b","c"),("c","d"),("b","e"),("e","f")],size=(10,7))


if __name__ == "__main__":
    main()
