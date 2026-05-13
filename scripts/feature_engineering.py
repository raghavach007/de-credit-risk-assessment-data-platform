
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

applications = pd.read_csv(BASE_DIR / "raw_data/applications/applications.csv")
bureau = pd.read_csv(BASE_DIR / "raw_data/bureau/bureau_data.csv")
transactions = pd.read_csv(BASE_DIR / "raw_data/transactions/transactions.csv")
income = pd.read_csv(BASE_DIR / "raw_data/income_verification/income_verification.csv")
alt = pd.read_csv(BASE_DIR / "raw_data/alt_data/alt_data.csv")

txn_features = transactions.groupby("applicant_id").agg({
    "avg_balance": "mean",
    "overdraft_count": "sum",
    "salary_credit_regular": "mean",
    "num_transactions": "mean"
}).reset_index()

bureau_features = bureau.groupby("applicant_id").agg({
    "credit_score": "mean",
    "payment_history_pct": "mean",
    "credit_utilisation_pct": "mean",
    "derogatory_marks": "sum",
    "missed_payments_12m": "sum"
}).reset_index()

income_features = income.groupby("applicant_id").agg({
    "verified_income": "mean",
    "employment_tenure_yrs": "mean",
    "itr_filed": "max"
}).reset_index()

alt_features = alt.groupby("applicant_id").agg({
    "utility_on_time_pct": "mean",
    "rental_on_time_pct": "mean",
    "telecom_on_time_pct": "mean",
    "alt_score": "mean"
}).reset_index()

df = applications.merge(bureau_features, on="applicant_id", how="left")
df = df.merge(txn_features, on="applicant_id", how="left")
df = df.merge(income_features, on="applicant_id", how="left")
df = df.merge(alt_features, on="applicant_id", how="left")

df.fillna(0, inplace=True)

df["income_to_loan_ratio"] = df["declared_income"] / (df["loan_amount"] + 1)
df["risk_flag"] = df["decision"].apply(lambda x: 1 if x == "Declined" else 0)

output_path = BASE_DIR / "outputs/processed_features.csv"
df.to_csv(output_path, index=False)

print("Feature engineering completed.")
print(f"Saved at: {output_path}")
