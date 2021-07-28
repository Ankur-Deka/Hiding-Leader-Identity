import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time

# dataset
import os
from torch.utils.data import Dataset, DataLoader, random_split

torch.manual_seed(1)

# Parameters
nAgents = 6
obsDim = 2
inpDim = nAgents*obsDim
hiddenDim = 2*nAgents
outDim = nAgents
initSteps = 10
batch_size = 20
val_frac = 0.2



class TrajDataset(Dataset):
    # swarm trajectory dataset                                      
    def __init__(self, root_dir, num_agents, obs_dim, init_steps, mode = 'HDD'):
        self.root_dir = root_dir
        self.files = [os.path.join(self.root_dir,f) for f in os.listdir(self.root_dir)]
        self.num_files = len(self.files)
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.init_steps = init_steps
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
        
        temp = X[:,i*self.obs_dim:(i+1)*self.obs_dim]
        X[:,i*self.obs_dim:(i+1)*self.obs_dim] = X[:,0:self.obs_dim]
        X[:,0:self.obs_dim] = temp
        
        Y[:] = i
        Y = Y[self.init_steps:]
        
        X,Y = torch.tensor(X, dtype = torch.float32),torch.tensor(Y, dtype = torch.int64)
        return(X,Y)
        

# Setup model
class adversaryNet(nn.Module):
    def __init__(self, inpDim, hiddenDim, outDim, initSteps):
        super(adversaryNet, self).__init__()
        self.inpDim = inpDim
        self.hiddenDim = hiddenDim
        self.outDim = outDim
        self.initSteps = initSteps                     
        self.lstm = nn.LSTM(inpDim, hiddenDim)           # Input dim is 3, output dim is 3
        self.fc1 = nn.Linear(hiddenDim, outDim)
        
    def forward(self,x):  # x in shape [batch_size, seq_len, inp_dim]
        batchSize, seqLen, _ = x.shape
        
        # reshape,feed to lstm
        out = x.transpose(0,1)                           # reshape for lstm [seq_len, batch_size, inp_dim]
        initData, data = out[:self.initSteps], out[self.initSteps:]  # initialization data and actual data to generate output
        _, hidden = self.lstm(initData)                  # initialize the hidden states with some data
        out, _ = self.lstm(data)                         # get actual output to be use for prediction
        
        # reshape and pass through fcn
        out = out.transpose(0,1).contiguous().view(-1,self.hiddenDim)    # [(batch_size)*(seqLen-initSteps)) X hiddenDim]
        out = self.fc1(out)                                              # [(batch_size)*(seqLen-initSteps)) X outDim]
        
        # reshape and return
        out = out.view(batchSize, seqLen-self.initSteps,self.outDim) # batch_size x (seqLen-initSteps) X outDim
        return(out)

def loss_fn(outputs,labels,criterion):
    _,_,outDim = outputs.shape
    loss = criterion(outputs.contiguous().view(-1,outDim), labels.contiguous().view(-1))
    return(loss)


device = 'cuda:0'
adversary = adversaryNet(inpDim=inpDim, hiddenDim=hiddenDim, outDim=outDim, initSteps=initSteps).to(device)

# dataset
traj_dataset = TrajDataset('out_files',6,2,10, mode = 'HDD')
dataset_size = len(traj_dataset)
val_size = int(val_frac*dataset_size)
train_size = dataset_size - val_size
train_dataset, val_dataset = random_split(traj_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)
train_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=10)

# optimizer
criterion = nn.CrossEntropyLoss()
learning_rate = 1e-4
optimizer = torch.optim.Adam(adversary.parameters(), lr=learning_rate)
epochs = 10


# train
print('starting')
t_start = time.time()
for e in range(epochs):
    loss_epoch = 0
    for X,Y in train_loader:
        outputs = adversary(X.to(device))
        loss = loss_fn(outputs.to(device), Y.to(device), criterion)
        loss.backward()
        optimizer.step()
        loss_epoch += loss.data
    print('loss epoch {}'.format(loss_epoch))
t_end = time.time()
print('time taken {}'.format(t_end-t_start))