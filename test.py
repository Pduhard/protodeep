import numpy as np

a = np.array([
    [0, 1],
    [1, 2]
])
b = np.array([
    [0, 1],
    [1, 2]
])

print(np.dot(a, b) + [2, 3])
print(np.dot([0, 1], b))
print(np.dot([1,2], b) + [2, 3])