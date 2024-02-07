import numpy as np
import matplotlib.pyplot as plt
import torch

arr = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
rows = torch.arange(arr.shape[0]).reshape(-1,1)
cols = torch.tensor([2,1,0]).reshape(-1,1)
print(arr)
print(arr[rows, cols])
print(arr.sum().item())