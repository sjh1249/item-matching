import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime

# Step 1: Load the CO2 and purchase data
print("Loading datasets...")
co2_data = pd.read_csv("20250501_carbon_data.csv")
purchase_data = pd.read_csv("attr_all.csv", dtype=str, encoding="ISO-8859-1", engine="python", on_bad_lines='skip')
print("Datasets loaded successfully.")

# Step 2: Load pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully.")

# Step 3: Clean data
co2_data['product_long_description'] = co2_data['product_long_description'].fillna("").astype(str)
purchase_data['long_desc'] = purchase_data['long_desc'].fillna("").astype(str)

# Step 4: Remove rows with missing or empty 'product' field
purchase_data = purchase_data[purchase_data['product'].notna() & (purchase_data['product'].str.strip() != '')].copy()

# Step 5: Create a lookup dictionary for direct matching
co2_lookup = co2_data.set_index('product_id')[['product_long_description', 'mean_GHG', 'mean_Biodiversity']].to_dict(orient='index')

# Step 6: Precompute CO2 embeddings
print("Generating CO2 embeddings...")
co2_embeddings = model.encode(
    co2_data['product_long_description'].tolist(), 
    convert_to_numpy=True, 
    normalize_embeddings=True
).astype(np.float32)
print(f"CO2 embeddings generated: {co2_embeddings.shape}")

# Step 7: Match items
def match_purchase_items():
    matches = []

    for i, row in purchase_data.iterrows():
        product_name = row['product']
        long_desc = row['long_desc']
        print(f"Processing {i+1}/{len(purchase_data)}: {product_name}")

        if product_name in co2_lookup:
            # Exact match
            match_info = co2_lookup[product_name]
            matches.append({
                'product': product_name,
                'purchase_item': long_desc,
                'best_match': match_info['product_long_description'],
                'co2_value': match_info['mean_GHG'],
                'bio_value': match_info['mean_Biodiversity'],
                'similarity_score': 1.0,
                'match_type': 'exact'
            })
        else:
            # Semantic match
            purchase_embedding = model.encode([long_desc], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
            similarity = np.dot(co2_embeddings, purchase_embedding[0])
            best_match_idx = np.argmax(similarity)
            matches.append({
                'product': product_name,
                'purchase_item': long_desc,
                'best_match': co2_data.iloc[best_match_idx]['product_long_description'],
                'co2_value': co2_data.iloc[best_match_idx]['mean_GHG'],
                'bio_value': co2_data.iloc[best_match_idx]['mean_Biodiversity'],
                'similarity_score': similarity[best_match_idx],
                'match_type': 'semantic'
            })
    
    return matches

# Step 8: Run matching
print("Matching purchase items...")
matches = match_purchase_items()
print("All matches completed.")

# Step 9: Create DataFrame and save
matches_df = pd.DataFrame(matches)
filename = f"{datetime.today().strftime('%Y%m%d')}_matched_mandala.csv"
matches_df.to_csv(filename, index=False)
print(f"Results saved to {filename}")

# Step 10: Display sample
print(matches_df.head())
