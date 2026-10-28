
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from perceptron_model import Perceptron
from logistic_model import LogisticRegression


# 1. Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "spambase.data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# 2. Load dataset
data = pd.read_csv(DATA_PATH, header=None)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

print("Dataset shape:", data.shape)
print("Number of features:", X.shape[1])


# 3. Split data (same split for both models)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 4. Scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train_array = y_train.to_numpy()


# 5. Train Perceptron
perceptron = Perceptron(
    learning_rate=0.01,
    n_iterations=1000
)

perceptron.fit(X_train_scaled, y_train_array)


# 6. Train Logistic Regression
logistic_model = LogisticRegression(
    learning_rate=0.01,
    n_iterations=1000
)

logistic_model.fit(X_train_scaled, y_train_array)


# 7. Evaluate both models
models = {
    "Perceptron": perceptron,
    "Logistic Regression": logistic_model
}

for name, model in models.items():
    y_pred = model.predict(X_test_scaled)

    print(f"\n===== {name} =====")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("Classification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Not Spam", "Spam"],
            zero_division=0
        )
    )


# 8. Save both models and the shared scaler
joblib.dump(perceptron, MODELS_DIR / "perceptron.pkl")
joblib.dump(logistic_model, MODELS_DIR / "logistic_regression.pkl")
joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

print("\nBoth models and the scaler were saved successfully!")
print("Location:", MODELS_DIR)
