import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv("products.csv")

# ─────────────────────────────────────────────
# USER PROFILE
# ─────────────────────────────────────────────
user_profile = {
    "community": "Muslim",
    "preferred_categories": ["Ethnic Wear", "Beauty", "Religious"],
    "price_range": (0, 1500),
    "past_purchases": ["abaya", "hijab", "halal", "prayer"]
}

# ─────────────────────────────────────────────
# RECOMMENDATION ENGINE
# ─────────────────────────────────────────────
def recommend(user_profile, df, top_n=10):
    # Step 1: Filter by community
    community_df = df[df["community"] == user_profile["community"]].copy()

    if community_df.empty:
        return pd.DataFrame(), []

    # Step 2: Filter by price range
    min_price, max_price = user_profile["price_range"]
    community_df = community_df[
        (community_df["price"] >= min_price) &
        (community_df["price"] <= max_price)
    ]

    if community_df.empty:
        return pd.DataFrame(), []

    # Step 3: Content-based filtering using TF-IDF on tags
    past_purchases_text = " ".join(user_profile["past_purchases"])
    preferred_categories_text = " ".join(user_profile["preferred_categories"])
    user_text = past_purchases_text + " " + preferred_categories_text

    # Combine tags and category for each product
    community_df["combined_features"] = community_df["tags"] + " " + community_df["category"]

    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    all_texts = list(community_df["combined_features"]) + [user_text]
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Cosine similarity between user and each product
    user_vector = tfidf_matrix[-1]
    product_vectors = tfidf_matrix[:-1]
    similarity_scores = cosine_similarity(user_vector, product_vectors).flatten()

    community_df["relevance_score"] = similarity_scores

    # Step 4: Rank by rating + relevance score (weighted)
    community_df["final_score"] = (
        0.5 * community_df["relevance_score"] +
        0.5 * (community_df["rating"] / 5.0)
    )

    # Step 5: Category boost — preferred categories get higher score
    def category_boost(row):
        if row["category"] in user_profile["preferred_categories"]:
            return row["final_score"] * 1.2
        return row["final_score"]

    community_df["final_score"] = community_df.apply(category_boost, axis=1)

    # Sort and get top N
    top_products = community_df.sort_values("final_score", ascending=False).head(top_n)

    # Step 6: Generate explanations
    explanations = []
    for _, row in top_products.iterrows():
        reasons = []
        if row["category"] in user_profile["preferred_categories"]:
            reasons.append(f"matches your preferred category '{row['category']}'")
        if row["rating"] >= 4.7:
            reasons.append(f"highly rated ({row['rating']}/5)")
        if row["price"] <= 500:
            reasons.append("budget friendly")
        tag_list = [t.strip() for t in row["tags"].split(",")]
        matching_tags = [t for t in tag_list if t in user_profile["past_purchases"]]
        if matching_tags:
            reasons.append(f"similar to your past purchases ({', '.join(matching_tags)})")
        explanation = f"Recommended because it " + (", ".join(reasons) if reasons else "fits your community profile")
        explanations.append(explanation)

    return top_products.reset_index(drop=True), explanations


# ─────────────────────────────────────────────
# RUN & DISPLAY
# ─────────────────────────────────────────────
top_products, explanations = recommend(user_profile, df)

print("=" * 70)
print(f"TOP 10 RECOMMENDATIONS FOR {user_profile['community'].upper()} USER")
print("=" * 70)

for i, (_, row) in enumerate(top_products.iterrows()):
    print(f"\n#{i+1}  {row['name']}")
    print(f"     Category : {row['category']}")
    print(f"     Price    : ₹{row['price']}")
    print(f"     Rating   : {row['rating']}/5")
    print(f"     Score    : {round(row['final_score'], 3)}")
    print(f"     Why      : {explanations[i]}")

print("\n" + "=" * 70)
print("Recommendation engine working successfully!")
