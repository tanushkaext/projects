import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

# 1. Load dataset
df = pd.read_csv("Indian Sign Language Gesture Landmarks.csv")
X = df.drop(columns=["target"])
y = df["target"]

LETTERS = [chr(ord('A') + i) for i in range(26)]
n_classes = 26

# 2. Time-block split (avoids data leakage)
train_frames, test_frames = [], []
for label in sorted(y.unique()):
    class_df = df[df["target"] == label]
    split_point = int(len(class_df) * 0.8)
    train_frames.append(class_df.iloc[:split_point])
    test_frames.append(class_df.iloc[split_point:])

train_df = pd.concat(train_frames)
test_df = pd.concat(test_frames)

X_train = train_df.drop(columns=["target"])
y_train = train_df["target"]
X_test = test_df.drop(columns=["target"])
y_test = test_df["target"]

y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))

# 3. Define both models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "MLP (Neural Network)": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42),
}

# 4. Train, evaluate, and plot ROC for each model
for name, model in models.items():
    print("\n" + "=" * 60)
    print(f"MODEL: {name}")
    print("=" * 60)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --- Accuracy ---
    acc = accuracy_score(y_test, y_pred)
    print(f"\nOverall Accuracy: {acc*100:.2f}%\n")

    # --- Precision, Recall, F1 only (no support column) ---
    print("Classification Report (Precision, Recall, F1-score per letter):")
    report_dict = classification_report(y_test, y_pred, target_names=LETTERS, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df = report_df.drop(columns=["support"])
    print(report_df.round(3))

    # --- ROC Curve (macro + per-letter) ---
    y_score = model.predict_proba(X_test)

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)

    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")

    # Macro-average ROC plot
    plt.figure(figsize=(7, 7))
    plt.plot(all_fpr, mean_tpr, color="navy", linewidth=2,
             label=f"Macro-average ROC (AUC = {macro_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name} (Macro-Average)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"roc_macro_{safe_name}.png", dpi=150)
    plt.close()

    # Per-letter ROC plot
    plt.figure(figsize=(10, 10))
    colors = cycle(plt.cm.tab20.colors)
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, linewidth=1,
                 label=f"{LETTERS[i]} (AUC={roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves - {name} (Per Letter, One-vs-Rest)")
    plt.legend(loc="lower right", fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(f"roc_per_letter_{safe_name}.png", dpi=150)
    plt.close()

    print(f"\nMacro AUC: {macro_auc:.3f}")
    print(f"Saved: roc_macro_{safe_name}.png and roc_per_letter_{safe_name}.png")

print("\nDone. All reports printed above, all ROC images saved in this folder.")