from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

# This might take a minute the first time as it downloads the model (approx 50MB)
print("Loading the embedding model... (this might take a moment)")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Define some words to test
sentences = ["Dog", "Puppy", "Cat", "Refrigerator"]

# 3. Convert words to Vectors (Numbers)
print("Turning words into numbers...")
vectors = embeddings.embed_documents(sentences)

# 4. Let's look at the data
print(f"\nWe turned 'Dog' into a list of {len(vectors[0])} numbers.")
print(f"Here are the first 5 numbers for 'Dog': {vectors[0][:5]}")

# 5. The Magic: Calculate Similarity
# We use 'Cosine Similarity' - a math trick to see how close two vectors are.
# 1.0 means identical. 0.0 means totally different.

def cosine_similarity(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

similarity_dog_puppy = cosine_similarity(vectors[0], vectors[1]) # Dog vs Puppy
similarity_dog_fridge = cosine_similarity(vectors[0], vectors[3]) # Dog vs Refrigerator

print(f"\nSimilarity between Dog and Puppy: {similarity_dog_puppy:.4f} (Should be high)")
print(f"Similarity between Dog and Refrigerator: {similarity_dog_fridge:.4f} (Should be low)")