import numpy as np
import torch as T 

arr = np.array([2, 1, 0, 0, 0 , 2, 0])
w = np.where(arr==0)[0][0]

print(w)