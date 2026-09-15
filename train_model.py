"""
Trains a placement prediction model on the real dataset
(student_placement_prediction_dataset_2026.csv, saved here as placement_data.csv).

Saves:
- model.pkl
- scaler.pkl
- feature_names.pkl
- encoders.pkl   (LabelEncoders for categorical columns)
"""
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("placement_data.csv")

# Drop columns that aren't useful / would leak the answer
# - student_id: just an identifier
# - salary_package_lpa: only known AFTER placement, so it's a leak -> drop for the classifier
df = df.drop(columns=["student_id", "salary_package_lpa"])

TARGET = "placement_status"

# Encode target: Placed -> 1, Not Placed -> 0
df[TARGET] = df[TARGET].map({"Placed": 1, "Not Placed": 0})

# Categorical columns to encode
CATEGORICAL = ["gender", "branch", "college_tier", "volunteer_experience"]
encoders = {}
for col in CATEGORICAL:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

FEATURES = [c for c in df.columns if c != TARGET]

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "LogisticRegression": LogisticRegression(max_iter=2000),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
}

best_model, best_name, best_acc = None, None, -1
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    print(f"\n{name} accuracy: {acc:.3f}")
    print(classification_report(y_test, preds))
    if acc > best_acc:
        best_model, best_name, best_acc = model, name, acc

print(f"\nBest model: {best_name} ({best_acc:.3f} accuracy)")
print("Confusion matrix:\n", confusion_matrix(y_test, best_model.predict(X_test_scaled)))

# Feature importance / coefficients (helps understand what drives placement)
if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nTop features:\n", importances.head(10))
elif hasattr(best_model, "coef_"):
    importances = pd.Series(best_model.coef_[0], index=FEATURES).sort_values(key=abs, ascending=False)
    print("\nTop features (by coefficient magnitude):\n", importances.head(10))

with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("feature_names.pkl", "wb") as f:
    pickle.dump(FEATURES, f)
with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

print("\nSaved model.pkl, scaler.pkl, feature_names.pkl, encoders.pkl")
