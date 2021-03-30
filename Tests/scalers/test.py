from sklearn import preprocessing
import numpy as np
X_train = np.array([[ 1., -1.,  2.],
[ 2.,  0.,  0.],
[ 0.,  1., -1.]])
scaler = preprocessing.StandardScaler().fit(X_train)
print(scaler)

print(scaler.mean_)

print(scaler.scale_)

X_scaled = scaler.transform(X_train)
print(X_scaled)
print(scaler)

X_scaled = scaler.transform(X_train)
print(X_scaled)