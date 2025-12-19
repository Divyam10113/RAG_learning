import numpy as np

# DATA STORE
# [Is Round?, Is Food?]
apple      = np.array([0.9, 0.9])
basketball = np.array([0.9, 0.1])
banana     = np.array([0.2, 0.95])

# THE USER QUERY
# "I want a healthy snack" (Low roundness, High food)
query = np.array([0.1, 0.8])

def cosine_similarity(vec_a, vec_b):
    return np.dot(vec_a, vec_b)/(np.linalg.norm(vec_a)*np.linalg.norm(vec_b))

applesim = cosine_similarity(apple, query)
basketballsim = cosine_similarity(basketball, query)
bananasim = cosine_similarity(banana, query)

print(applesim)
print(basketballsim)
print(bananasim)