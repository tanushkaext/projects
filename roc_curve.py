import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle

# Load data and model
df = pd.read_csv("Indian Sign Language Gesture Landmarks.csv")
model = joblib.load("isl_model.pkl")

X = df.drop(columns=["target"])
y = df["target"]

# Same time-block split as training/evaluation
train_frames, test_frames = [], []
for label in sorted(y.unique()):
    class_df = df[df["target"] == label]
    split_point = int(len(class_df) * 0.8)
    test_frames.append(class_df.iloc[split_point:])

test_df = pd.concat(test_frames)
X_test = test_df.drop(columns=["target"])
y_test = test_df["target"]

LETTERS = [chr(ord('A') + i) for i in range(26)]
n_classes = 26

# Binarize labels for one-vs-rest ROC (needed for multi-class ROC)
y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))

# Get predicted probabilities for each class
y_score = model.predict_proba(X_test)

# Compute ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute macro-average ROC curve
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes
macro_auc = auc(all_fpr, mean_tpr)

# ---- Plot 1: Macro-average ROC curve (clean summary, good for report) ----
plt.figure(figsize=(7, 7))
plt.plot(all_fpr, mean_tpr, color="navy", linewidth=2,
         label=f"Macro-average ROC (AUC = {macro_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - ISL Alphabet Recognition (Macro-Average)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve_macro.png", dpi=150)
plt.show()
print("Saved macro-average ROC curve as roc_curve_macro.png")

# ---- Plot 2: Per-letter ROC curves (detailed, optional for appendix) ----
plt.figure(figsize=(10, 10))
colors = cycle(plt.cm.tab20.colors)
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, linewidth=1,
             label=f"{LETTERS[i]} (AUC={roc_auc[i]:.2f})")
plt.plot([0, 1], [0, 1], "k--", linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Per Letter (One-vs-Rest)")
plt.legend(loc="lower right", fontsize=6, ncol=2)
plt.tight_layout()
plt.savefig("roc_curve_per_letter.png", dpi=150)
plt.show()
print("Saved per-letter ROC curves as roc_curve_per_letter.png")