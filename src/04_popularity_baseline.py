
from pathlib import Path
import pandas as pd

# Load cleaned interactions
df = pd.read_csv(
    "data/cleaned/electronics_cleaned.csv"
)

# Calculate product statistics
products = df.groupby("parent_asin").agg(
    average_rating=("rating", "mean"),
    rating_count=("rating", "count")
).reset_index()

# Global mean rating
C = df["rating"].mean()

# Minimum-vote threshold
m = products["rating_count"].quantile(0.90)

# Weighted rating score
v = products["rating_count"]
R = products["average_rating"]

products["weighted_score"] = (
    (v / (v + m)) * R
    + (m / (v + m)) * C
)

# Sort by weighted popularity
products = products.sort_values(
    "weighted_score",
    ascending=False
)

print("\n===== TOP 10 ELECTRONICS PRODUCTS =====")
print(
    products[
        ["parent_asin", "average_rating",
         "rating_count", "weighted_score"]
    ].head(10).to_string(index=False)
)

print(f"\nGlobal average rating: {C:.4f}")
print(f"Minimum-vote threshold (90th percentile): {m:.2f}")

# Save results
output_dir = Path("results")
output_dir.mkdir(parents=True, exist_ok=True)

products.to_csv(
    output_dir / "popularity_rankings.csv",
    index=False
)

print("\nPopularity rankings saved successfully!")
