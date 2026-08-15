import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load data and model
df = pd.read_csv("Indian Sign Language Gesture Landmarks.csv")
model = joblib.load("isl_model.pkl")

X = df.drop(columns=["target"])
y = df["target"]

# Same time-block split as training, so this matches your real test set
train_frames, test_frames = [], []
for label in sorted(y.unique()):
    class_df = df[df["target"] == label]
    split_point = int(len(class_df) * 0.8)
    test_frames.append(class_df.iloc[split_point:])

test_df = pd.concat(test_frames)
X_test = test_df.drop(columns=["target"])
y_test = test_df["target"]

y_pred = model.predict(X_test)

LETTERS = [chr(ord('A') + i) for i in range(26)]

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LETTERS)

fig, ax = plt.subplots(figsize=(12, 12))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=True)
plt.title("ISL Alphabet Recognition - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("Saved as confusion_matrix.png")