import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def recommend_products(user_profile, csv_path="data/products.csv", top_n=10):
    """
    Recommend products based on user profile using TF-IDF + cosine similarity.
    """

    # ─────────────────────────────────────────────
    # LOAD DATA
    # ─────────────────────────────────────────────
    df = pd.read_csv('products.csv')

    # ─────────────────────────────────────────────
    # STEP 1: FILTER BY COMMUNITY
    # ─────────────────────────────────────────────
    community = user_profile.get("community", "")
    df = df[df["community"] == community].copy()

    if df.empty:
        return pd.DataFrame(), []

    # ─────────────────────────────────────────────
    # STEP 2: FILTER BY PRICE
    # ─────────────────────────────────────────────
    min_price, max_price = user_profile.get("price_range", (0, float("inf")))
    df = df[(df["price"] >= min_price) & (df["price"] <= max_price)]

    if df.empty:
        return pd.DataFrame(), []

    # ─────────────────────────────────────────────
    # STEP 3: BUILD USER PROFILE TEXT
    # ─────────────────────────────────────────────
    past_purchases = " ".join(user_profile.get("past_purchases", []))
    preferred_categories = " ".join(user_profile.get("preferred_categories", []))

    user_text = past_purchases + " " + preferred_categories

    # Combine product features
    df["combined_features"] = df["tags"] + " " + df["category"]

    # ─────────────────────────────────────────────
    # STEP 4: TF-IDF + COSINE SIMILARITY
    # ─────────────────────────────────────────────
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(list(df["combined_features"]) + [user_text])

    similarity_scores = cosine_similarity(
        tfidf_matrix[-1], tfidf_matrix[:-1]
    ).flatten()

    df["relevance_score"] = similarity_scores

    # ─────────────────────────────────────────────
    # STEP 5: FINAL SCORING
    # ─────────────────────────────────────────────
    df["final_score"] = (
        0.5 * df["relevance_score"] +
        0.5 * (df["rating"] / 5.0)
    )

    # Category boost
    preferred_cats = user_profile.get("preferred_categories", [])

    df["final_score"] = df.apply(
        lambda row: row["final_score"] * 1.2
        if row["category"] in preferred_cats
        else row["final_score"],
        axis=1
    )

    # ─────────────────────────────────────────────
    # STEP 6: SORT & SELECT TOP N
    # ─────────────────────────────────────────────
    top_products = df.sort_values("final_score", ascending=False).head(top_n)

    # ─────────────────────────────────────────────
    # STEP 7: EXPLANATIONS
    # ─────────────────────────────────────────────
    explanations = []

    for _, row in top_products.iterrows():
        reasons = []

        if row["category"] in preferred_cats:
            reasons.append(f"matches preferred category '{row['category']}'")

        if row["rating"] >= 4.7:
            reasons.append(f"highly rated ({row['rating']}/5)")

        if row["price"] <= 500:
            reasons.append("budget friendly")

        # FIX: tags split by space (not comma)
        tags = row["tags"].split()
        matches = [t for t in tags if t in user_profile.get("past_purchases", [])]

        if matches:
            reasons.append(f"similar to past purchases ({', '.join(matches)})")

        explanation = "Recommended because it " + (
            ", ".join(reasons) if reasons else "fits your profile"
        )

        explanations.append(explanation)

    return top_products.reset_index(drop=True), explanations


# ─────────────────────────────────────────────
# TEST RUN (SAFE)
# ─────────────────────────────────────────────
if __name__ == "__main__":

    user_profile = {
        "community": "Muslim",
        "preferred_categories": ["Ethnic", "Beauty", "Religious"],  # FIXED
        "price_range": (0, 1500),
        "past_purchases": ["abaya", "hijab", "halal", "prayer"]
    }

    products, explanations = recommend_products(user_profile)

    print("=" * 70)
    print(f"TOP {len(products)} RECOMMENDATIONS FOR {user_profile['community'].upper()} USER")
    print("=" * 70)

    for i, (_, row) in enumerate(products.iterrows()):
        print(f"\n#{i+1}  {row['name']}")
        print(f"     Category : {row['category']}")
        print(f"     Price    : ₹{row['price']}")
        print(f"     Rating   : {row['rating']}/5")
        print(f"     Score    : {round(row['final_score'], 3)}")
        print(f"     Why      : {explanations[i]}")

    print("\n" + "=" * 70)
    print("Recommendation engine working successfully!")