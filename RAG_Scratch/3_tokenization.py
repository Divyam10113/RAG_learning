import re

# 1. THE DATA
text = "Artificial Intelligence is fascinating because it allows computers to learn. Machine Learning is a subset of AI that focuses on data."

# 2. YOUR LOGIC: Splitting by words and special characters
# This regex pattern finds words (\w+) OR non-alphanumeric characters like punctuation ([^\w\s])
tokens = re.findall(r'\w+|[^\w\s]', text)

print(f"Token List: {tokens[:10]}...") # Printing first 10 to see the split
print(f"Total Tokens: {len(tokens)}")
print("-" * 30)

# 3. HYPERPARAMETERS (Now in 'words', not characters)
chunk_size = 6  # Grab 6 tokens at a time
overlap = 2     # Keep 2 tokens context
start = 0

# 4. THE LOOP (Applied to the list)
while start < len(tokens):
    # We slice the LIST now, not the string
    window = tokens[start : start + chunk_size]
    
    # Join them back into a string just for pretty printing
    print(f"Window: {window}")
    
    # Stop if we are at the end (same logic as before)
    if start + chunk_size >= len(tokens):
        break
        
    start += (chunk_size - overlap)