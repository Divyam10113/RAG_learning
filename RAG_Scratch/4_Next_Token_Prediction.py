import numpy as np

token_ids = [2, 5, 12, 10, 8, 13, 1, 4] 

# 1. Determine dimensions
# We need columns up to index 13, so size must be 14
vocab_size = max(token_ids) + 1 
num_rows = len(token_ids)

# 2. Create the blank canvas (The Matrix)
# Shape: (8 rows, 14 columns)
one_hot_matrix = np.zeros((num_rows, vocab_size))

# 3. Flip the bits
for row_index in range(num_rows):
    # Get the ID for the current word (e.g., 2)
    col_index = token_ids[row_index]
    
    # Go to that specific row and column, and set to 1
    one_hot_matrix[row_index, col_index] = 1

print("Shape:", one_hot_matrix.shape)
print(one_hot_matrix)

embed_dim = 3
weighted_encoding = []
weighted_matrix = np.random.rand(vocab_size, embed_dim)
for i in one_hot_matrix:
    weighted_encoding.append(np.matmul(i, weighted_matrix))
print(weighted_encoding)