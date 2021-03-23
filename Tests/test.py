import numpy as np
import time

start = time.time()
for i in range(100):
    a = np.ones(10000000)
print(f'tt: {time.time() - start}')


a = np.empty(10000000)
start = time.time()
for i in range(100):
    # a -= a
    a.fill(i)
print(f'tt: {time.time() - start}')
