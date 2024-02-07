import numpy as np
import torch
import matplotlib.pyplot as plt


cp = torch.load('Data/checkpoint1.pth')
# torch.save(cp['buffer'],'Data/buffer11.pth')
del cp['buffer']
torch.save(cp, 'Data/checkpoint11.pth')


cp = torch.load('Data/checkpoint2.pth')
# torch.save(cp['buffer'],'Data/buffer12.pth')
del cp['buffer']
torch.save(cp, 'Data/checkpoint12.pth')
