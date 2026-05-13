
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(BASE_DIR / "outputs/processed_features.csv")

categorical_cols = ["product_type", "loan_purpose", "employment_type", "employer_category", "city"]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

drop_cols = [
    "application_id",
    "applicant_id",
    "application_date",
    "decision_date",
    "decision"
]

X = df.drop(columns=drop_cols + ["risk_flag"])
y = df["risk_flag"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

pred_probs = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, pred_probs)

print(f"AUC Score: {auc:.4f}")

joblib.dump(model, BASE_DIR / "outputs/trained_model.pkl")

print("Model saved successfully.")
