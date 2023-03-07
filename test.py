import numpy as np
from sklearn.metrics import confusion_matrix


true = np.array([0, 1, 1, 1, 1, 0, 0, 0, 1])
pred = np.array([0, 1, 1, 0, 0, 1, 0, 0, 1])

print(confusion_matrix(true, pred, labels=[1, 0]))