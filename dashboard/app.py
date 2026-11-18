
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="ElectroRec AI | Recommendation System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cleaned"
RESULTS = ROOT / "results"

# ==========================================
# CUSTOM DESIGN
# ==========================================

st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_data():

    metadata = pd.read_csv(
        DATA / "product_metadata.csv"
    )

    popularity = pd.read_csv(
        RESULTS / "popularity_rankings.csv"
    )

    item_cf = pd.read_csv(
        RESULTS / "collaborative_recommendations.csv"
    )

    svd = pd.read_csv(
        RESULTS / "svd_recommendations.csv"
    )

    comparison = pd.read_csv(
        RESULTS / "model_comparison.csv"
    )

    for df in [metadata, popularity, item_cf, svd]:
        df["parent_asin"] = (
            df["parent_asin"].astype(str)
        )

    metadata = metadata.drop_duplicates(
        subset="parent_asin"
    )

    return metadata, popularity, item_cf, svd, comparison


try:
    metadata, popularity, item_cf, svd, comparison = load_data()
except (FileNotFoundError, ValueError) as error:
    st.error(f"Could not load project data: {error}")
    st.stop()

# ==========================================
# HELPERS
# ==========================================

def enrich_products(df):
    return df.merge(
        metadata,
        on="parent_asin",
        how="left",
        suffixes=("", "_meta")
    )


def display_products(df, score_column=None, limit=12):

    if df.empty:
        st.info("No products available for this selection.")
        return

    df = df.head(limit)

    for start in range(0, len(df), 4):
        cols = st.columns(4, gap="medium")

        for col, (_, product) in zip(
            cols,
            df.iloc[start:start + 4].iterrows()
        ):
            with col:
                with st.container(border=True):

                    image = product.get("image_url")

                    if pd.notna(image) and str(image).startswith("http"):
                        st.image(image, use_container_width=True)
                    else:
                        st.markdown("📦 Image unavailable")

                    title = product.get("title")

                    if pd.isna(title) or not str(title).strip():
                        title = "Unknown product"

                    title = str(title)

                    st.markdown(
                        f"**{title[:85]}{'...' if len(title) > 85 else ''}**"
                    )

                    rating = product.get("average_rating")

                    if pd.notna(rating):
                        st.caption(f"⭐ {rating} / 5")

                    price = product.get("price")

                    if pd.notna(price) and str(price).strip():
                        st.write(f"Price: {price}")

                    if score_column and score_column in product.index:
                        score = pd.to_numeric(
                            product[score_column],
                            errors="coerce"
                        )

                        if pd.notna(score):
                            st.caption(
                                f"Recommendation score: {score:.4f}"
                            )

                    asin = product["parent_asin"]

                    st.link_button(
                        "View on Amazon",
                        f"https://www.amazon.com/dp/{asin}",
                        use_container_width=True
                    )

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("⚡ ElectroRec ")
st.sidebar.caption("Intelligent Electronics Recommendation System")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "🛍️ Product Catalog",
        "🎯 Recommendations",
        "🔍 Customer Insights",
        "📊 Model Analytics"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("**Recommendation Models**")
st.sidebar.write("Popularity-Based")
st.sidebar.write("Item-Based Collaborative Filtering")
st.sidebar.write("Matrix Factorization (SVD)")

st.sidebar.divider()
st.sidebar.caption("DSA 4060 | Recommender Systems")

# ==========================================
# OVERVIEW
# ==========================================

if page == "🏠 Overview":

    st.markdown(
    '<div class="main-title">⚡ ElectroRec AI</div>',
    unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Discover products powered by machine learning.'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Products", f"{len(metadata):,}")

    all_users = set(item_cf["user_id"].astype(str))
    all_users.update(svd["user_id"].astype(str))

    c2.metric("Users with Recommendations", f"{len(all_users):,}")
    c3.metric("Recommendation Models", "3")
    c4.metric("Products with Image URLs",
              f"{metadata['image_url'].notna().sum():,}")

    st.divider()

    st.subheader("🔥 Popular Products")

    popular_products = enrich_products(
        popularity.sort_values(
            "weighted_score",
            ascending=False
        )
    )

    display_products(
        popular_products,
        score_column="weighted_score",
        limit=8
    )

    st.divider()

    st.subheader("About the System")

    st.write(
        "SmartCart AI is a machine-learning-based e-commerce "
        "recommendation system developed using Amazon Electronics "
        "review data. It compares popularity-based recommendations, "
        "item-based collaborative filtering, and SVD matrix "
        "factorization to recommend relevant products."
    )

# ==========================================
# PRODUCT CATALOG
# ==========================================

elif page == "🛍️ Product Catalog":

    st.title("🛍️ Explore Products")

    st.caption(
        "Browse Amazon Electronics products from the project dataset."
    )

    search = st.text_input(
        "Search products",
        placeholder="Search headphones, cables, tablets..."
    )

    catalog = metadata.copy()

    if search:
        catalog = catalog[
            catalog["title"]
            .fillna("")
            .str.contains(search, case=False, regex=False)
        ]

    if "main_category" in catalog.columns:

        categories = sorted(
            catalog["main_category"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        category = st.selectbox(
            "Category",
            ["All Categories"] + categories
        )

        if category != "All Categories":
            catalog = catalog[
                catalog["main_category"] == category
            ]

    st.write(f"Showing {len(catalog):,} matching products")

    limit = st.select_slider(
        "Products to display",
        options=[8, 12, 20, 40, 80],
        value=20
    )

    display_products(catalog, limit=limit)

# ==========================================
# PERSONALIZED RECOMMENDATIONS
# ==========================================

elif page == "🎯 Recommendations":

    st.title("🎯 Personalized Recommendations")

    st.write(
        "Select a user and recommendation algorithm "
        "to view predicted product preferences."
    )

    model = st.selectbox(
        "Recommendation Algorithm",
        [
            "SVD",
            "Item-Based Collaborative Filtering",
            "Popularity-Based"
        ]
    )

    if model == "SVD":
        source = svd
        score_column = "svd_score"

    elif model == "Item-Based Collaborative Filtering":
        source = item_cf
        score_column = "recommendation_score"

    else:
        source = popularity
        score_column = "weighted_score"

    if model != "Popularity-Based":

        users = sorted(
            source["user_id"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_user = st.selectbox(
            "Select User ID",
            users
        )

        recommendations = source[
            source["user_id"].astype(str) == selected_user
        ].copy()

        st.info(
            f"Displaying {model} recommendations "
            f"for user {selected_user}."
        )

    else:
        recommendations = source.copy()

        st.info(
            "Popularity-based recommendations are the same "
            "for every user."
        )

    recommendations[score_column] = pd.to_numeric(
        recommendations[score_column],
        errors="coerce"
    )

    recommendations = recommendations.sort_values(
        score_column,
        ascending=False
    )

    recommendations = enrich_products(recommendations)

    top_n = st.slider(
        "Number of recommendations",
        min_value=4,
        max_value=20,
        value=10
    )

    display_products(
        recommendations,
        score_column=score_column,
        limit=top_n
    )

    with st.expander("View recommendation data"):
        st.dataframe(
            recommendations.head(top_n),
            use_container_width=True
        )

# ==========================================
# MODEL ANALYTICS
# ==========================================


# ==========================================
# CUSTOMER INSIGHTS & MODEL COMPARISON
# ==========================================

elif page == "🔍 Customer Insights":

    st.title("🔍 Customer Insights & Model Comparison")

    st.write(
        "Explore a customer's previous ratings and compare "
        "personalized recommendations from Item-CF and SVD."
    )

    @st.cache_data
    def load_training_history():
        return pd.read_csv(
            DATA / "train.csv",
            dtype={"user_id": str, "parent_asin": str}
        )

    history = load_training_history()

    # Only users with recommendations from both models
    item_users = set(item_cf["user_id"].astype(str))
    svd_users = set(svd["user_id"].astype(str))

    common_users = sorted(item_users & svd_users)

    if not common_users:
        st.warning(
            "No common users found between Item-CF and SVD."
        )
        st.stop()

    selected_user = st.selectbox(
        "Select Customer",
        common_users,
        key="insights_user"
    )

    # ======================================
    # CUSTOMER RATING HISTORY
    # ======================================

    user_history = history[
        history["user_id"] == selected_user
    ].copy()

    user_history = user_history.merge(
        metadata[
            ["parent_asin", "title", "image_url"]
        ],
        on="parent_asin",
        how="left"
    )

    st.subheader("Customer Rating History")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Products Rated",
        len(user_history)
    )

    c2.metric(
        "Average Rating",
        f"{user_history['rating'].mean():.2f} / 5"
        if not user_history.empty else "N/A"
    )

    c3.metric(
        "Highly Rated Products",
        int((user_history["rating"] >= 4).sum())
    )

    if not user_history.empty:

        st.dataframe(
            user_history[
                ["title", "rating", "parent_asin"]
            ].sort_values(
                "rating",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    # ======================================
    # PERSONALIZED RECOMMENDATIONS
    # ======================================

    st.subheader("Recommendations for This Customer")

    top_n = st.slider(
        "Number of products per model",
        min_value=1,
        max_value=10,
        value=5,
        key="insights_top_n"
    )

    cf_recs = item_cf[
        item_cf["user_id"].astype(str) == selected_user
    ].copy()

    svd_recs = svd[
        svd["user_id"].astype(str) == selected_user
    ].copy()

    cf_recs = cf_recs.sort_values(
        "recommendation_score",
        ascending=False
    )

    svd_recs = svd_recs.sort_values(
        "svd_score",
        ascending=False
    )

    cf_recs = enrich_products(cf_recs)
    svd_recs = enrich_products(svd_recs)

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("### Item-Based Collaborative Filtering")

        st.caption(
            "Recommends products based on similarity "
            "to products the customer rated highly."
        )

        for _, product in cf_recs.head(top_n).iterrows():

            with st.container(border=True):
                image_col, info_col = st.columns([1, 2])

                with image_col:
                    if pd.notna(product.get("image_url")):
                        st.image(
                            product["image_url"],
                            use_container_width=True
                        )

                with info_col:
                    st.markdown(
                        f"**{str(product.get('title', 'Unknown'))[:90]}**"
                    )
                    st.caption(
                        f"Similarity score: "
                        f"{product['recommendation_score']:.4f}"
                    )

    with right:
        st.markdown("### SVD Matrix Factorization")

        st.caption(
            "Ranks unseen products using learned "
            "user-product preference patterns."
        )

        for _, product in svd_recs.head(top_n).iterrows():

            with st.container(border=True):
                image_col, info_col = st.columns([1, 2])

                with image_col:
                    if pd.notna(product.get("image_url")):
                        st.image(
                            product["image_url"],
                            use_container_width=True
                        )

                with info_col:
                    st.markdown(
                        f"**{str(product.get('title', 'Unknown'))[:90]}**"
                    )
                    st.caption(
                        f"SVD score: {product['svd_score']:.4f}"
                    )

    st.divider()

    # ======================================
    # RECOMMENDATION OVERLAP
    # ======================================

    st.subheader("Recommendation Overlap")

    cf_items = set(
        cf_recs.head(top_n)["parent_asin"]
    )

    svd_items = set(
        svd_recs.head(top_n)["parent_asin"]
    )

    common_items = cf_items & svd_items

    overlap_percentage = (
        len(common_items) / top_n * 100
    )

    m1, m2 = st.columns(2)

    m1.metric(
        "Products Recommended by Both",
        len(common_items)
    )

    m2.metric(
        "Top-N Recommendation Overlap",
        f"{overlap_percentage:.1f}%"
    )

    if common_items:
        st.success(
            "Both algorithms recommended some of "
            "the same products."
        )
    else:
        st.info(
            "The algorithms recommended different "
            "products for this customer."
        )

# ==========================================
# MODEL ANALYTICS
# ==========================================



elif page == "📊 Model Analytics":

    st.title("📊 Model Performance Analytics")

    st.write(
        "Compare recommendation quality across "
        "Popularity, Item-CF, and SVD."
    )

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    metrics = [
        "Precision@10",
        "Recall@10",
        "Coverage@10",
        "Novelty@10"
    ]

    available_metrics = [
        metric for metric in metrics
        if metric in comparison.columns
    ]

    selected_metric = st.selectbox(
        "Select Evaluation Metric",
        available_metrics
    )

    if selected_metric:

        fig = px.bar(
            comparison,
            x="Model",
            y=selected_metric,
            color="Model",
            text_auto=".4f",
            title=f"{selected_metric} by Recommendation Model"
        )

        fig.update_layout(
            showlegend=False,
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    st.subheader("Evaluation Metrics Explained")

    with st.expander("Precision@10"):
        st.write(
            "Measures the proportion of recommended products "
            "that are relevant to the user."
        )

    with st.expander("Recall@10"):
        st.write(
            "Measures the proportion of relevant test products "
            "successfully recommended."
        )

    with st.expander("Coverage@10"):
        st.write(
            "Measures how much of the product catalog "
            "the recommendation model can recommend."
        )

    with st.expander("Novelty@10"):
        st.write(
            "Measures how uncommon the recommended products are."
        )

    st.subheader("Model Comparison Summary")

    if "Recall@10" in comparison.columns:

        best_recall = comparison.loc[
            comparison["Recall@10"].idxmax(),
            "Model"
        ]

        st.success(
            f"Highest Recall@10: {best_recall}"
        )

    if "Coverage@10" in comparison.columns:

        best_coverage = comparison.loc[
            comparison["Coverage@10"].idxmax(),
            "Model"
        ]

        st.info(
            f"Highest Coverage@10: {best_coverage}"
        )


