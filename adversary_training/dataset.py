import os
import numpy as np
import torch
from torch.utils.data import Dataset


# -------------------- format -------------------- #
# input: [batch_size, num_steps, num_agents*obs_dim]
# output: [batch_size, num_steps-init_steps, num_agents*obs_dim]  

class TrajDataset(Dataset):
    # swarm trajectory dataset                                      
    def __init__(self, root_dir, num_agents, obs_dim, init_steps, stacked = False, mode = 'HDD'):
        self.root_dir = root_dir
        self.files = [os.path.join(self.root_dir,f) for f in os.listdir(self.root_dir)]
        self.num_files = len(self.files)
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.init_steps = init_steps
        self.stacked = stacked
        self.mode = mode
        if self.mode == 'RAM':
            print('mode is ram')
            self.X = []
            self.Y = []
            for i in range(self.num_files):
                file = self.files[i]
                data = np.load(file)
                X,Y = data[:,:-1], data[:,-1]
                X,Y = self.transform(X,Y)
                self.X.append(X.view(1,51,12))
                self.Y.append(Y.view(1,-1))
            self.X = torch.from_numpy(np.vstack(self.X))
            self.Y = torch.from_numpy(np.vstack(self.Y))

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        if self.mode == 'RAM':
            return(self.X[idx], self.Y[idx])
        else:
            file = self.files[idx]
            data = np.load(file)
            X,Y = data[:,:-1], data[:,-1]
            X,Y = self.transform(X,Y)
            return X,Y
    
    def transform(self, X, Y):
        # select random location for leader
        i = np.random.randint(self.num_agents)
        temp = X[:,i*self.obs_dim:(i+1)*self.obs_dim].copy()
        X[:,i*self.obs_dim:(i+1)*self.obs_dim] = X[:,0:self.obs_dim].copy()
        X[:,0:self.obs_dim] = temp.copy()
        
        
        if self.stacked:
            # time_steps x num_agents x obs_dim
            X = X.reshape((-1, self.num_agents, self.obs_dim))
            Y = i
        else:    
            Y[:] = i
            Y = Y[self.init_steps:]


        X,Y = torch.tensor(X, dtype = torch.float32),torch.tensor(Y, dtype = torch.int64)
        return(X,Y)