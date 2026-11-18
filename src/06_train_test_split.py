
from pathlib import Path
import pandas as pd

df = pd.read_csv(
    "data/cleaned/electronics_cleaned.csv"
)

# Order interactions chronologically for each user
df = df.sort_values(
    ["user_id", "timestamp", "parent_asin"]
).reset_index(drop=True)

# Position of each interaction in user history
df["position"] = df.groupby("user_id").cumcount()

# Number of interactions per user
df["user_count"] = df.groupby("user_id")[
    "user_id"
].transform("size")

# Reserve the final 20% for testing,
# ensuring at least one test interaction per user
df["test_count"] = (
    df["user_count"] * 0.20
).apply(lambda x: max(1, int(x)))

df["is_test"] = (
    df["position"] >=
    df["user_count"] - df["test_count"]
)

train = df[~df["is_test"]].copy()
test = df[df["is_test"]].copy()

# Remove temporary columns
columns = ["position", "user_count", "test_count", "is_test"]
train = train.drop(columns=columns)
test = test.drop(columns=columns)

output_dir = Path("data/cleaned")
output_dir.mkdir(parents=True, exist_ok=True)

train.to_csv(output_dir / "train.csv", index=False)
test.to_csv(output_dir / "test.csv", index=False)

print("\n===== CHRONOLOGICAL SPLIT =====")
print("Training interactions:", len(train))
print("Testing interactions:", len(test))
print("Total interactions:", len(train) + len(test))
print("Training users:", train["user_id"].nunique())
print("Testing users:", test["user_id"].nunique())
print("Training products:", train["parent_asin"].nunique())
print("Testing products:", test["parent_asin"].nunique())

unseen_items = (
    set(test["parent_asin"]) -
    set(train["parent_asin"])
)

print("Test products unseen in training:", len(unseen_items))
print("\nTrain/test files saved successfully!")
