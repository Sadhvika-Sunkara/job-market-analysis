"""
Static PNG charts for the Part 3 Medium article, built from the real
model_results.json / ablation_results.json numbers (no fabricated data).
Follows the dataviz skill's light-mode reference palette.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ---- palette (light mode, reference instance) ----
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
BLUE_LIGHT = "#9ec5f4"   # sequential step ~200
ORANGE = "#eb6834"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK_PRIMARY,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
})

results = json.load(open("model_results.json"))
ablation = json.load(open("ablation_results.json"))

# =====================================================================
# Chart 1: model comparison, 5-fold CV macro-F1, with error bars
# =====================================================================
cv = results["cv_results"]
order = ["MultinomialNB", "LogisticRegression", "RandomForest", "LinearSVM"]
labels = ["Naive Bayes", "Logistic\nRegression", "Random\nForest", "Linear SVM\n(selected)"]
means = [cv[k]["mean_macro_f1"] for k in order]
stds = [cv[k]["std_macro_f1"] for k in order]
colors = [BLUE_LIGHT, BLUE_LIGHT, BLUE_LIGHT, BLUE]

fig, ax = plt.subplots(figsize=(8, 5.2), dpi=200)
y = np.arange(len(order))
bars = ax.barh(y, means, xerr=stds, color=colors, height=0.55,
                edgecolor="none", capsize=4,
                error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1.3, "capthick": 1.3})
for yi, m in zip(y, means):
    ax.text(m + 0.025, yi, f"{m:.3f}", va="center", ha="left",
             color=INK_PRIMARY, fontsize=12, fontweight="bold")

ax.set_yticks(y, labels, fontsize=12)
ax.set_xlim(0, 0.95)
ax.set_xlabel("5-fold cross-validated macro-F1 (training split)", fontsize=11)
ax.set_title("Choosing a model: 4-class role classifier", fontsize=15, fontweight="bold",
             color=INK_PRIMARY, loc="left", pad=14)
ax.grid(axis="x", color=GRIDLINE, linewidth=1, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color(BASELINE)
ax.invert_yaxis()
fig.tight_layout()
fig.savefig("chart_model_comparison.png", dpi=200)
plt.close(fig)

# =====================================================================
# Chart 2: confusion matrix heatmap (held-out test set, Linear SVM)
# =====================================================================
cm = np.array(results["test_results"]["confusion_matrix"])
class_names = results["class_names"]
nice_names = {"ai_ml_engineer": "AI/ML\nEngineer", "data_analyst": "Data\nAnalyst",
              "data_engineer": "Data\nEngineer", "data_scientist": "Data\nScientist"}
tick_labels = [nice_names[c] for c in class_names]

row_sums = cm.sum(axis=1, keepdims=True)
cm_pct = cm / row_sums * 100

seq_ramp = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("seq_blue", seq_ramp)

fig, ax = plt.subplots(figsize=(7.6, 6.6), dpi=200)
im = ax.imshow(cm_pct, cmap=cmap, vmin=0, vmax=100)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        pct = cm_pct[i, j]
        txt_color = "white" if pct > 55 else INK_PRIMARY
        ax.text(j, i, f"{cm[i, j]}\n({pct:.0f}%)", ha="center", va="center",
                 fontsize=11, color=txt_color, fontweight="bold" if i == j else "normal")

ax.set_xticks(range(len(class_names)), tick_labels, fontsize=10.5)
ax.set_yticks(range(len(class_names)), tick_labels, fontsize=10.5)
ax.set_xlabel("Predicted role", fontsize=11, labelpad=10)
ax.set_ylabel("Actual role", fontsize=11, labelpad=10)
ax.set_title("Held-out test set: who gets confused with whom", fontsize=13.5,
             fontweight="bold", color=INK_PRIMARY, loc="left", pad=14)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0)
fig.tight_layout()
fig.savefig("chart_confusion_matrix.png", dpi=200)
plt.close(fig)

# =====================================================================
# Chart 3: 2016 vs. now — does the skill gap hold up?
# =====================================================================
skills = ["Machine\nLearning", "Excel", "Statistics", "Python", "SQL"]
gap_2016 = [46, 35, 9, 58, 6]          # magnitude of the DS-vs-DA gap, 2016 (95 postings)
gap_now = [49.6, 27.7, 21.9, 37.1, -17.7]  # signed: negative = analyst-favored / reversed
favor_2016 = ["DS", "DA", "DS", "DS", "DS"]
favor_now = ["DS", "DA", "DS", "DS", "DA"]

fig, ax = plt.subplots(figsize=(9, 5.6), dpi=200)
y = np.arange(len(skills))
h = 0.32

# plot as signed magnitude: positive = scientist-favored, negative = analyst-favored
signed_2016 = [g if f == "DS" else -g for g, f in zip(gap_2016, favor_2016)]
signed_now = [g if g >= 0 else g for g in gap_now]  # already signed correctly in list above,
# but 2016 list stored as unsigned magnitude + direction, so recompute cleanly:
signed_2016 = [46, -35, 9, 58, 6]
signed_now = [49.6, -27.7, 21.9, 37.1, -17.7]

bars1 = ax.barh(y + h/2, signed_2016, height=h, color=BLUE, label="2016 (95 postings)", zorder=3)
bars2 = ax.barh(y - h/2, signed_now, height=h, color=ORANGE, label="Now (7,523 postings)", zorder=3)

for yi, v in zip(y + h/2, signed_2016):
    ax.text(v + (2 if v >= 0 else -2), yi, f"{v:+.0f}", va="center",
             ha="left" if v >= 0 else "right", fontsize=10, color=INK_PRIMARY)
for yi, v in zip(y - h/2, signed_now):
    ax.text(v + (2 if v >= 0 else -2), yi, f"{v:+.1f}", va="center",
             ha="left" if v >= 0 else "right", fontsize=10, color=INK_PRIMARY)

ax.axvline(0, color=BASELINE, linewidth=1.4, zorder=2)
ax.set_yticks(y, skills, fontsize=12)
ax.set_xlabel("Data Scientist − Data Analyst mention-rate gap (percentage points)\n"
              "← more common in Data Analyst postings   |   more common in Data Scientist postings →",
              fontsize=9.5, color=INK_SECONDARY)
ax.set_title("Does the 2016 signal hold up at scale?", fontsize=15, fontweight="bold",
             color=INK_PRIMARY, loc="left", pad=14)
ax.grid(axis="x", color=GRIDLINE, linewidth=1, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.set_xlim(-70, 70)
ax.legend(loc="lower right", frameon=False, fontsize=10.5, labelcolor=INK_SECONDARY)
ax.invert_yaxis()
fig.tight_layout()
fig.savefig("chart_2016_vs_now.png", dpi=200)
plt.close(fig)

print("Saved: chart_model_comparison.png, chart_confusion_matrix.png, chart_2016_vs_now.png")
