from src.recommendation.recommender import recommend_products

user = {
    "community": "Muslim",
    "preferred_categories": ["Ethnic", "Beauty", "Religious"],
    "price_range": (0, 1500),
    "past_purchases": ["abaya", "hijab", "halal", "prayer"]
}

# ✅ UNPACK RESULT
products, explanations = recommend_products(user)

# Optional: keep only clean columns
products = products[['name', 'category', 'price', 'rating', 'final_score']]

# ✅ CLEAN DISPLAY
print("=" * 70)
print(f"TOP {len(products)} RECOMMENDATIONS FOR {user['community'].upper()} USER")
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