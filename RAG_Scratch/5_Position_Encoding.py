import numpy as np

# 1. Setup Dimensions
seq_len = 5   # We have 5 words
d_model = 4   # Each word is a vector of size 4

# 2. Fake Word Embeddings (Just random numbers for now)
# Shape: (5, 4)
embeddings = np.random.rand(seq_len, d_model)

# 3. Create a placeholder Position Matrix
# Shape: (5, 4)
position_matrix = np.zeros((seq_len, d_model))

# --- YOUR TASK START ---
# Loop through each row (position) from 0 to seq_len
# Calculate val = np.sin(row_index)
# Fill the ENTIRE row with that value
# Hint: position_matrix[i, :] = ...

for i in range(seq_len):
    # Write code here to fill the row
    pass 

# --- YOUR TASK END ---

# 4. The Final Step: Add them together!
final_input = embeddings + position_matrix

print("Original Embedding (Row 0):\n", embeddings[0])
print("Position Value (Row 0):\n", position_matrix[0])
print("Final Sum (Row 0):\n", final_input[0])