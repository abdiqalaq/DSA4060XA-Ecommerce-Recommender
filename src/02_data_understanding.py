
import pandas as pd

# Load dataset
df = pd.read_csv(
    "data/raw/electronics_reviews_sample.csv"
)

print("\n===== AMAZON ELECTRONICS DATASET =====")

# Dataset dimensions
print("\nTotal interactions:", len(df))
print("Unique users:", df["user_id"].nunique())
print("Unique products:", df["parent_asin"].nunique())

# Missing values
print("\nMissing values:")
print(df.isnull().sum())

# Duplicate user-product interactions
duplicates = df.duplicated(
    subset=["user_id", "parent_asin"]
).sum()

print("\nDuplicate user-product pairs:", duplicates)

# Rating distribution
print("\nRating distribution:")
print(df["rating"].value_counts().sort_index())

# User-item matrix sparsity
users = df["user_id"].nunique()
products = df["parent_asin"].nunique()

unique_pairs = df.drop_duplicates(
    subset=["user_id", "parent_asin"]
).shape[0]

sparsity = 1 - (
    unique_pairs / (users * products)
)

print(f"\nUser-item matrix sparsity: {sparsity:.4%}")

# Most active users
print("\nTop 5 most active users:")
print(df["user_id"].value_counts().head())

# Most reviewed products
print("\nTop 5 most reviewed products:")
print(df["parent_asin"].value_counts().head())

print("\n===== ANALYSIS COMPLETE =====")
