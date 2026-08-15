import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load the dataset
df = pd.read_csv("Indian Sign Language Gesture Landmarks.csv")

# 2. Separate features (X) and label (y)
X = df.drop(columns=["target"])
y = df["target"]

# 3. Split into train and test sets by time-block per class
# (avoids leakage from near-identical adjacent video frames)
train_frames = []
test_frames = []

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

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}\n")

# 4. Define the models we want to compare
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="rbf", random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "MLP (Neural Network)": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42),
}

results = {}
trained_models = {}

# 5. Train and evaluate each model
for name, model in models.items():
    print(f"Training {name}...")
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    results[name] = {"accuracy": accuracy, "train_time": train_time}
    trained_models[name] = model

    print(f"  Accuracy: {accuracy * 100:.2f}%  |  Training time: {train_time:.1f}s\n")

# 6. Print a summary comparison table
print("=" * 50)
print(f"{'Model':<25}{'Accuracy':<15}{'Train Time (s)'}")
print("=" * 50)
for name, res in results.items():
    print(f"{name:<25}{res['accuracy']*100:.2f}%{'':<9}{res['train_time']:.1f}")

# 7. Pick the best model and print its detailed report
best_model_name = max(results, key=lambda k: results[k]["accuracy"])
best_model = trained_models[best_model_name]

print(f"\nBest model: {best_model_name} ({results[best_model_name]['accuracy']*100:.2f}% accuracy)")
print("\nDetailed classification report for best model:")
y_pred_best = best_model.predict(X_test)
print(classification_report(y_test, y_pred_best))

# 8. Save the best model for later use in real-time prediction
joblib.dump(best_model, "isl_model.pkl")
print(f"\nBest model ({best_model_name}) saved as isl_model.pkl")