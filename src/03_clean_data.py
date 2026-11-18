
from pathlib import Path
import pandas as pd

# Load original sample
df = pd.read_csv(
    "data/raw/electronics_reviews_sample.csv"
)

print("Original interactions:", len(df))

# Ensure correct data types
df["rating"] = pd.to_numeric(
    df["rating"], errors="coerce"
)

df["timestamp"] = pd.to_numeric(
    df["timestamp"], errors="coerce"
)

# Remove missing and invalid records
df = df.dropna(
    subset=["user_id", "parent_asin",
            "rating", "timestamp"]
)

df = df[df["rating"].between(1, 5)]

# Keep latest interaction for duplicate user-item pairs
df = df.sort_values("timestamp")
df = df.drop_duplicates(
    subset=["user_id", "parent_asin"],
    keep="last"
)

# Iterative 5-core filtering
while True:
    before = len(df)

    user_counts = df["user_id"].value_counts()
    df = df[
        df["user_id"].map(user_counts) >= 5
    ]

    item_counts = df["parent_asin"].value_counts()
    df = df[
        df["parent_asin"].map(item_counts) >= 5
    ]

    if len(df) == before:
        break

print("\n===== CLEANED DATASET =====")
print("Interactions:", len(df))
print("Users:", df["user_id"].nunique())
print("Products:", df["parent_asin"].nunique())

if len(df) > 0:
    users = df["user_id"].nunique()
    products = df["parent_asin"].nunique()

    sparsity = 1 - len(df) / (users * products)
    print(f"Sparsity: {sparsity:.4%}")

    output = Path("data/cleaned")
    output.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        output / "electronics_cleaned.csv",
        index=False
    )

    print("Cleaned dataset saved successfully!")
else:
    print("No interactions remain after filtering.")
    print("We need a larger or differently sampled dataset.")
