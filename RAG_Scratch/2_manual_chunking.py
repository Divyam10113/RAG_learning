# THE DATA
text = "Artificial Intelligence is fascinating because it allows computers to learn. Machine Learning is a subset of AI that focuses on data."

# HYPERPARAMETERS
chunk_size = 20  # The Window
overlap = 5      # The Safety Net

# YOUR TASK: Write the Loop
# Hint: Use a 'start' variable that increases by (chunk_size - overlap) each time.
# Hint: text[start : start + chunk_size]
start = 0
length = len(text)

print(f"Text Length: {length}")

# Write your while loop here...
while start <= length-chunk_size+1:
  print(text[start: start+chunk_size])
  start = start + chunk_size-overlap
