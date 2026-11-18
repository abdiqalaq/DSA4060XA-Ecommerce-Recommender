
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

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
N_COMPONENTS = 30
RANDOM_STATE = 42

print("Loading training dataset...", flush=True)

train = pd.read_csv(DATA / "train.csv")

# ==========================================
# 3. ENCODE USERS AND PRODUCTS
# ==========================================

user_codes, user_ids = pd.factorize(
    train["user_id"]
)

item_codes, item_ids = pd.factorize(
    train["parent_asin"]
)

# ==========================================
# 4. BUILD USER-ITEM MATRIX
# ==========================================

matrix = csr_matrix(
    (
        train["rating"].astype(float),
        (user_codes, item_codes)
    ),
    shape=(len(user_ids), len(item_ids))
)

print(f"Users: {len(user_ids):,}")
print(f"Products: {len(item_ids):,}")
print(f"Interactions: {len(train):,}")

# ==========================================
# 5. TRAIN MATRIX FACTORIZATION MODEL
# ==========================================

print("\nTraining TruncatedSVD model...", flush=True)

n_components = min(
    N_COMPONENTS,
    min(matrix.shape) - 1
)

if n_components < 1:
    raise ValueError(
        "Not enough users or products to train SVD."
    )

model = TruncatedSVD(
    n_components=n_components,
    random_state=RANDOM_STATE
)

user_factors = model.fit_transform(matrix)
item_factors = model.components_.T

print("\n===== MATRIX FACTORIZATION =====")
print("User factors shape:", user_factors.shape)
print("Item factors shape:", item_factors.shape)
print(
    "Explained variance:",
    round(
        model.explained_variance_ratio_.sum(),
        4
    )
)

# ==========================================
# 6. PREDICT USER-PRODUCT SCORES
# ==========================================

predicted_scores = (
    user_factors @ item_factors.T
)

# ==========================================
# 7. RECOMMENDATION FUNCTION
# ==========================================

def recommend_products(user_id, top_n=TOP_N):

    if user_id not in user_ids:
        raise ValueError("User not found in training data")

    user_index = user_ids.get_loc(user_id)

    scores = predicted_scores[
        user_index
    ].copy()

    # Exclude already rated products
    rated_items = matrix.getrow(
        user_index
    ).indices

    scores[rated_items] = -np.inf

    # Keep only valid scores
    ranked = np.argsort(scores)[::-1]
    ranked = ranked[np.isfinite(scores[ranked])]

    top_indices = ranked[:top_n]

    return pd.DataFrame({
        "parent_asin": item_ids[top_indices],
        "svd_score": scores[top_indices]
    })

# ==========================================
# 8. SELECT 100 ACTIVE USERS
# ==========================================

example_users = (
    train["user_id"]
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
# 9. GENERATE RECOMMENDATIONS
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
# 10. SAVE RESULTS
# ==========================================

output_file = RESULTS / "svd_recommendations.csv"

if results:
    final_df = pd.concat(
        results,
        ignore_index=True
    )
else:
    final_df = pd.DataFrame(
        columns=[
            "parent_asin",
            "svd_score",
            "user_id"
        ]
    )

final_df.to_csv(
    output_file,
    index=False
)

# ==========================================
# 11. FINAL SUMMARY
# ==========================================

print("\n===== SVD RESULTS =====")
print(f"Users processed: {len(example_users)}")
print(f"Users with recommendations: {len(results)}")
print(f"Recommendations generated: {len(final_df)}")
print(f"Saved to: {output_file}")
print("\nSVD recommendations saved successfully!")
