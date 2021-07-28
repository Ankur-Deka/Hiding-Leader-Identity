import sys
sys.path.append('../mape')
import torch
import torch.nn as nn
import numpy as np
from rlcore.distributions import Categorical


dist_class = Categorical()
dist = dist_class(torch.tensor([[-10,0,3.0]]).cuda())
print(dist.sample().cpu().numpy())
print(dist.mode().cpu().numpy())