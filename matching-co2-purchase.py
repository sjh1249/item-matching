import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime

# Step 1: Load the CO2 and purchase data
print("Loading datasets...")
co2_data = pd.read_csv("all_productenv.csv", sep="\t", dtype=str, encoding="ISO-8859-1")
purchase_data = pd.read_csv("kfdb_matches_wide.csv", dtype=str, header=0)
print("Datasets loaded successfully.")

# Step 2: Load pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully.")

# Step 3: Ensure no NaN values in text fields
co2_data['product_long_description'] = co2_data['product_long_description'].fillna("").astype(str)
co2_data['product_description'] = co2_data['product_description'].fillna("").astype(str)
purchase_data['kantar_productname'] = purchase_data['kantar_productname'].fillna("").astype(str)

# Step 4: Get embeddings for both datasets (convert to float16 to save memory)
print("Generating embeddings...")
co2_embeddings = model.encode(co2_data['product_long_description'] + " " + co2_data['product_description']).astype(np.float16)
print(f"CO2 embeddings generated: {co2_embeddings.shape}")

# Step 5: Match each purchase item one at a time (no memory issues)
def match_purchase_items():
    matches = []
    for i, purchase_item in enumerate(purchase_data['kantar_productname']):
        print(f"Processing {i+1}/{len(purchase_data)}: {purchase_item}")

        # Encode only one row at a time (no large memory usage)
        purchase_embedding = model.encode([purchase_item]).astype(np.float16)

        # Compute cosine similarity for this one row
        similarity = cosine_similarity(purchase_embedding, co2_embeddings)

        # Find best match
        best_match_idx = np.argmax(similarity)
        best_match_name = co2_data.iloc[best_match_idx]['product_long_description']
        co2_value = co2_data.iloc[best_match_idx]['mean_ghg']
        similarity_score = similarity[0][best_match_idx]

        # Store result immediately (no memory accumulation)
        matches.append({
            'purchase_item': purchase_item,
            'best_match': best_match_name,
            'co2_value': co2_value,
            'similarity_score': similarity_score
        })
    
    return matches

print("Matching purchase items...")
matches = match_purchase_items()
print("All matches found successfully.")

# Step 6: Convert results to DataFrame
matches_df = pd.DataFrame(matches, columns=['purchase_item', 'best_match', 'co2_value', 'similarity_score'])

# Step 7: Save to CSV with timestamp
today_date = datetime.today().strftime('%Y%m%d')
filename = f"{today_date}_matched.csv"
matches_df.to_csv(filename, index=False)
print(f"Results saved to {filename}")

# Step 8: Display first few results
print(matches_df.head())
print("The matches have been displayed successfully.")
