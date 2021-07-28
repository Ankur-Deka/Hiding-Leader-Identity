import torch
import torch.nn as nn


# Simple LSTM model
class adversaryNetV0(nn.Module):
    def __init__(self, inpDim, hiddenDim, outDim, initSteps):
        super(adversaryNetV0, self).__init__()
        self.inpDim = inpDim
        self.hiddenDim = hiddenDim
        self.outDim = outDim
        self.initSteps = initSteps                     
        self.lstm1 = nn.LSTM(inpDim, hiddenDim)           # Input dim is 3, output dim is 3
        self.lstm2 = nn.LSTM(hiddenDim, hiddenDim)
        self.fc1 = nn.Linear(hiddenDim, outDim)
        
    def forward(self,x):  # x in shape [batch_size, seq_len, inp_dim]
        batchSize, seqLen, _ = x.shape
        
        # reshape,feed to lstm
        out = x.transpose(0,1)                           # reshape for lstm [seq_len, batch_size, inp_dim]
        initData, data = out[:self.initSteps], out[self.initSteps:]  # initialization data and actual data to generate output
        out, h1 = self.lstm1(initData)                  # initialize the hidden states with some data
        # _, h2 = self.lstm2(out)
        
        out, _ = self.lstm1(data, h1)                         # get actual output to be use for prediction
        # out, _ = self.lstm2(out, h2)
        
        # reshape and pass through fcn
        out = out.transpose(0,1).contiguous().view(-1,self.hiddenDim)    # [(batch_size)*(seqLen-initSteps)) X hiddenDim]
        out = self.fc1(out)                                              # [(batch_size)*(seqLen-initSteps)) X outDim]
        
        
        # reshape and return
        out = out.view(batchSize, seqLen-self.initSteps,self.outDim) # batch_size x (seqLen-initSteps) X outDim
        return(out)

# Scalable LSTM model
class adversaryNetV1(nn.Module):
    def __init__(self, inpDim, hiddenDim, initSteps):
        super(adversaryNetV1, self).__init__()
        self.inpDim = inpDim 								# corresponding to single agent
        self.hiddenDim = hiddenDim
        self.initSteps = initSteps                     
        self.lstm1 = nn.LSTM(inpDim, hiddenDim)           # Input dim is 3, output dim is 3
        # self.lstm2 = nn.LSTM(hiddenDim, hiddenDim)
        self.fc1 = nn.Linear(hiddenDim, hiddenDim)
        # self.fc2 = nn.Linear(hiddenDim, outDim)
        self.leaderEmbed = nn.Parameter(torch.ones(hiddenDim))

    def forward(self,x):  # x in shape [batch_size, seq_len, inp_dim*num_agents]
        batchSize, seqLen, D = x.shape
        numAgents = D//self.inpDim

        # reshape,feed to lstm
        out = x.transpose(0,1)                           # reshape for lstm [seq_len, batch_size, num_agents*inp_dim]

        out = out.contiguous().view(seqLen, batchSize*numAgents, -1)						 # reshape to [seq_len, batch_size*num_agents, inp_dim]

        # -------- initilize lstm states -------- #
        initData, data = out[:self.initSteps], out[self.initSteps:]  # initialization data and actual data to generate output
        _, h1 = self.lstm1(initData)                  # initialize the hidden states with some data
        
        # -------- get actual output -------- #
        out, _ = self.lstm1(data, h1)                         # get actual output to be use for prediction, out = [seqLen-initSteps, batchSize*numAgents, hiddenDim]
        
        # -------- pass through fcn -------- #
        out = out.contiguous().view(-1, self.hiddenDim) 		# [(seqLen-initSteps)*batchSize*numAgents, hiddenDim]
        out = self.fc1(out)										# [(seqLen-initSteps)*batchSize*numAgents, hiddenDim]

        # -------- dot product with adversary embedding -------- #
        out = torch.matmul(out, self.leaderEmbed)					# [(seqLen-initSteps)*batchSize*numAgents]
        out = out.view(-1, batchSize, numAgents).transpose(0,1)	# [batchSize, (seqLen-initSteps), numAgents] # don't apply softmax as it'll be applied by cross entropy loss

        return(out)

# AdversaryNetV2 code taken from
# https://github.com/proroklab/private_flocking
class adversaryNetV2(nn.Module):

    def __init__(self, inpDims, hiddenCh, hiddenDim, outDim, applyMaxPool=False):
        super(adversaryNetV2, self).__init__()
        self.inpDim1 = inpDims[0] 
        self.inpDim2 = inpDims[1]
        self.inpCh = inpDims[2]
        self.hiddenCh = hiddenCh
        self.hiddenDim = hiddenDim
        self.outDim = outDim
        self.applyMaxPool = applyMaxPool
        self.kernSize = 3

        self.feature = nn.Sequential(
            nn.Conv2d(self.inpCh, self.hiddenCh, kernel_size=self.kernSize, stride=1, padding=1),
            nn.BatchNorm2d(self.hiddenCh),
            nn.ReLU(inplace=True)
        )

        self.maxPool = nn.MaxPool2d(kernel_size=self.kernSize, stride=1, padding=1)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.inpDim1 * self.inpDim2 * self.hiddenCh, self.hiddenDim),
            nn.ReLU(),
            nn.Linear(self.hiddenDim, self.outDim)
        )

    def forward(self, x):
        x = self.feature(x)
        if self.applyMaxPool:
        	x = self.maxPool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Scalable LSTM model with self attention
class adversaryNetV3(nn.Module):
    def __init__(self, inpDim, hiddenDim, initSteps, normDotProd):
        super(adversaryNetV3, self).__init__()
        self.inpDim = inpDim                                # corresponding to single agent
        self.hiddenDim = hiddenDim
        self.initSteps = initSteps                     
        self.normDotProd = normDotProd
        self.lstm1 = nn.LSTM(inpDim, hiddenDim)           # Input dim is 3, output dim is 3
        # self.lstm2 = nn.LSTM(hiddenDim, hiddenDim)
        self.fc1 = nn.Linear(hiddenDim, hiddenDim)
        if self.normDotProd:
            self.normalize = nn.functional.normalize
        # self.fc2 = nn.Linear(hiddenDim, outDim)

    def forward(self,x):  # x in shape [batch_size, seq_len, inp_dim*num_agents]
        batchSize, seqLen, D = x.shape
        numAgents = D//self.inpDim

        # reshape,feed to lstm
        out = x.transpose(0,1)                           # reshape for lstm [seq_len, batch_size, num_agents*inp_dim]
        out = out.contiguous().view(seqLen, batchSize*numAgents, -1)                         # reshape to [seq_len, batch_size*num_agents, inp_dim]

        # -------- initilize lstm states -------- #
        initData, data = out[:self.initSteps], out[self.initSteps:]  # initialization data and actual data to generate output
        _, h1 = self.lstm1(initData)                  # initialize the hidden states with some data
        
        # -------- get actual output -------- #
        out, _ = self.lstm1(data, h1)                         # get actual output to be use for prediction, out = [seqLen-initSteps, batchSize*numAgents, hiddenDim]
        
        # -------- pass through fcn -------- #
        out = out.contiguous().view(-1, self.hiddenDim)         # [(seqLen-initSteps)*batchSize*numAgents, hiddenDim]
        out = self.fc1(out)                                     # [(seqLen-initSteps)*batchSize*numAgents, hiddenDim]

        # -------- dot product between all pairs -------- #
        out = out.view(-1, numAgents, self.hiddenDim)    # [(seqLen-initSteps)*batchSize, numAgents, hiddenDim]
        if self.normDotProd:
            out = self.normalize(out,dim=2)
        # print('out', out[0,0])
        outT = out.transpose(1,2)                   #[(seqLen-initSteps)*batchSize, hiddenDim, numAgents]
        out = torch.matmul(out, outT)-torch.eye(numAgents).to(out.device)               # [(seqLen-initSteps)*batchSize, numAgents, numAgents]
        # print('out', out[0])
        out = -10*torch.mean(out, dim = 2)             # [(seqLen-initSteps)*batchSize, numAgents], as we want softMIN out here
        out = out.view(-1, batchSize, numAgents).transpose(0,1) # [batchSize, (seqLen-initSteps), numAgents] # don't apply softmax as it'll be applied by cross entropy loss

        return(out)