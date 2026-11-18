
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

K = 10

train = pd.read_csv("data/cleaned/train.csv")
test = pd.read_csv("data/cleaned/test.csv")

# Use training data only for all models
user_codes, user_ids = pd.factorize(train["user_id"])
item_codes, item_ids = pd.factorize(train["parent_asin"])

matrix = csr_matrix(
    (
        train["rating"].astype(float),
        (user_codes, item_codes)
    ),
    shape=(len(user_ids), len(item_ids))
)

# Model 1: Popularity
stats = train.groupby("parent_asin")["rating"].agg(
    ["mean", "count"]
)
C = train["rating"].mean()
m = stats["count"].quantile(0.90)

stats["score"] = (
    stats["count"] / (stats["count"] + m)
    * stats["mean"]
    + m / (stats["count"] + m) * C
)

pop_scores = (
    stats["score"]
    .reindex(item_ids)
    .to_numpy()
)

# Model 2: Item-based CF
similarity = cosine_similarity(matrix.T)
np.fill_diagonal(similarity, 0)

# Model 3: SVD
svd = TruncatedSVD(
    n_components=30,
    random_state=42
)
user_factors = svd.fit_transform(matrix)
item_factors = svd.components_.T

# Product popularity for novelty
item_counts = np.asarray(
    (matrix > 0).sum(axis=0)
).ravel()

n_users = matrix.shape[0]
n_items = matrix.shape[1]

novelty_values = -np.log2(
    np.maximum(item_counts, 1) / n_users
)

# Relevant held-out products (4 or 5 stars)
relevant = (
    test[test["rating"] >= 4]
    .groupby("user_id")["parent_asin"]
    .apply(set)
    .to_dict()
)

def top_k(scores, seen, k=10):
    scores = scores.copy()
    scores[seen] = -np.inf

    indices = np.argsort(scores)[::-1]
    indices = indices[np.isfinite(scores[indices])]

    return indices[:k]

def evaluate_model(model_name):
    precisions = []
    recalls = []
    novelties = []
    recommended_items = set()

    evaluated_users = 0

    for user_index, user_id in enumerate(user_ids):
        true_items = relevant.get(user_id, set())

        # Ignore relevant items unavailable in training
        true_items = true_items.intersection(
            set(item_ids)
        )

        if not true_items:
            continue

        seen = matrix.getrow(user_index).indices
        user_ratings = matrix.getrow(
            user_index
        ).toarray().ravel()

        if model_name == "Popularity":
            scores = pop_scores.copy()

        elif model_name == "Item-CF":
            liked = np.where(
                user_ratings >= 4,
                user_ratings,
                0
            )
            scores = similarity @ liked

        elif model_name == "SVD":
            scores = (
                user_factors[user_index]
                @ item_factors.T
            )

        else:
            raise ValueError("Unknown model")

        recommendations = top_k(scores, seen, K)

        predicted_items = set(
            item_ids[recommendations]
        )

        hits = len(predicted_items & true_items)

        precisions.append(hits / K)
        recalls.append(hits / len(true_items))

        recommended_items.update(recommendations)

        novelties.extend(
            novelty_values[recommendations]
        )

        evaluated_users += 1

    return {
        "Model": model_name,
        "Precision@10": np.mean(precisions),
        "Recall@10": np.mean(recalls),
        "Coverage@10": len(recommended_items) / n_items,
        "Novelty@10": np.mean(novelties),
        "Evaluated Users": evaluated_users
    }

print("\nEvaluating recommendation models...")

results = []

for model_name in ["Popularity", "Item-CF", "SVD"]:
    print(f"Evaluating {model_name}...")
    results.append(evaluate_model(model_name))

comparison = pd.DataFrame(results)

print("\n===== MODEL COMPARISON =====")
print(comparison.to_string(index=False))

output = Path("results")
output.mkdir(parents=True, exist_ok=True)

comparison.to_csv(
    output / "model_comparison.csv",
    index=False
)

print("\nModel comparison saved successfully!")
