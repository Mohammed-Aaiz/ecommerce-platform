import pandas as pd

def recommend_products(user_profile, csv_path="data/products.csv"):
    
    df = pd.read_csv(csv_path)

    # Clean text
    df['tags'] = df['tags'].astype(str)

    # Step 1: Community filter
    df = df[df['community'].str.lower() == user_profile['community'].lower()]

    # Step 2: Category filter
    df = df[df['category'].isin(user_profile['preferred_categories'])]

    # Step 3: Relevance score (simple keyword match)
    def score(row):
        score = 0
        for tag in user_profile.get("interests", []):
            if tag in row['tags']:
                score += 1
        return score

    df['relevance'] = df.apply(score, axis=1)

    # Step 4: Sort
    df = df.sort_values(by=['relevance', 'rating'], ascending=False)

    return df.head(10)[['name', 'price', 'rating', 'tags']].to_dict(orient="records")