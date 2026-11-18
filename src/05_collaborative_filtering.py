
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cleaned"
RESULTS = ROOT / "results"

RESULTS.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. CONFIGURATION
# ==========================================

TOP_N = 10
MAX_USERS = 100

print("Loading training dataset...", flush=True)

df = pd.read_csv(DATA / "train.csv")

# ==========================================
# 3. ENCODE USERS AND PRODUCTS
# ==========================================

user_codes, user_ids = pd.factorize(df["user_id"])
item_codes, item_ids = pd.factorize(df["parent_asin"])

# ==========================================
# 4. BUILD USER-ITEM MATRIX
# ==========================================

matrix = csr_matrix(
    (
        df["rating"].astype(float),
        (user_codes, item_codes)
    ),
    shape=(len(user_ids), len(item_ids))
)

print(f"Users: {len(user_ids):,}")
print(f"Products: {len(item_ids):,}")
print(f"Interactions: {len(df):,}")

# ==========================================
# 5. ITEM-ITEM COSINE SIMILARITY
# ==========================================

print("Computing item similarities...", flush=True)

item_similarity = cosine_similarity(
    matrix.T,
    dense_output=True
)

np.fill_diagonal(item_similarity, 0)

# ==========================================
# 6. RECOMMENDATION FUNCTION
# ==========================================

def recommend_products(user_id, top_n=TOP_N):

    if user_id not in user_ids:
        raise ValueError("User not found in training data")

    user_index = user_ids.get_loc(user_id)

    user_ratings = (
        matrix.getrow(user_index)
        .toarray()
        .ravel()
    )

    # Only use products rated 4 stars or higher
    liked_ratings = np.where(
        user_ratings >= 4,
        user_ratings,
        0
    )

    # Score products by similarity to liked items
    scores = item_similarity @ liked_ratings

    # Exclude products already rated
    scores[user_ratings > 0] = -np.inf

    # Exclude products without positive evidence
    scores[scores <= 0] = -np.inf

    ranked = np.argsort(scores)[::-1]
    ranked = ranked[np.isfinite(scores[ranked])]

    top_items = ranked[:top_n]

    return pd.DataFrame({
        "parent_asin": item_ids[top_items],
        "recommendation_score": scores[top_items]
    })

# ==========================================
# 7. SELECT 100 ACTIVE USERS
# ==========================================

example_users = (
    df["user_id"]
    .value_counts()
    .head(MAX_USERS)
    .index
)

print(
    f"\nGenerating Top-{TOP_N} recommendations "
    f"for {len(example_users)} users...",
    flush=True
)

# ==========================================
# 8. GENERATE RECOMMENDATIONS
# ==========================================

results = []

for i, user in enumerate(example_users, start=1):

    recommendations = recommend_products(
        user_id=user,
        top_n=TOP_N
    )

    if not recommendations.empty:
        recommendations["user_id"] = user
        results.append(recommendations)

    if i % 20 == 0:
        print(
            f"Processed {i}/{len(example_users)} users...",
            flush=True
        )

# ==========================================
# 9. SAVE RECOMMENDATIONS
# ==========================================

output_file = RESULTS / "collaborative_recommendations.csv"

if results:
    final_df = pd.concat(
        results,
        ignore_index=True
    )
else:
    final_df = pd.DataFrame(
        columns=[
            "parent_asin",
            "recommendation_score",
            "user_id"
        ]
    )

final_df.to_csv(
    output_file,
    index=False
)

# ==========================================
# 10. FINAL SUMMARY
# ==========================================

print("\n===== ITEM-CF RESULTS =====")
print(f"Users processed: {len(example_users)}")
print(f"Users with recommendations: {len(results)}")
print(f"Recommendations generated: {len(final_df)}")
print(f"Saved to: {output_file}")
print("\nItem-Based Collaborative Filtering completed!")
