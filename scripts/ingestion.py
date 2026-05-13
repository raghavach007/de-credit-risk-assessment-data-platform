
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

datasets = {
    "applications": BASE_DIR / "raw_data/applications/applications.csv",
    "bureau": BASE_DIR / "raw_data/bureau/bureau_data.csv",
    "transactions": BASE_DIR / "raw_data/transactions/transactions.csv",
    "income_verification": BASE_DIR / "raw_data/income_verification/income_verification.csv",
    "alt_data": BASE_DIR / "raw_data/alt_data/alt_data.csv"
}

def load_data():
    loaded = {}
    for name, path in datasets.items():
        loaded[name] = pd.read_csv(path)
        print(f"{name} loaded successfully -> {loaded[name].shape}")
    return loaded

if __name__ == "__main__":
    load_data()
