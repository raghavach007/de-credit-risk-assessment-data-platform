
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ks_2samp

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

model = joblib.load(BASE_DIR / "outputs/trained_model.pkl")

pred_probs = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, pred_probs)
gini = 2 * auc - 1

ks = ks_2samp(
    pred_probs[y_test == 0],
    pred_probs[y_test == 1]
).statistic

approval_rate = (df["decision"] == "Approved").mean() * 100

report = f'''
===== CREDIT RISK MODEL EVALUATION =====

AUC Score: {auc:.4f}
Gini Coefficient: {gini:.4f}
KS Statistic: {ks:.4f}
Approval Rate: {approval_rate:.2f}%
'''

print(report)

with open(BASE_DIR / "outputs/evaluation_metrics.txt", "w") as f:
    f.write(report)

print("Evaluation report generated.")
