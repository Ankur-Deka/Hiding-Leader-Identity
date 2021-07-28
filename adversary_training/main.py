# -------------------- generic -------------------- #
import argparse, os, sys, time, numpy as np
# -------------------- pytorch -------------------- #
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
# -------------------- defined -------------------- #
from models import *
from dataset import *
from utils import plot_trajectories, lossFunc
import pandas as pd

class AdversaryClass():
    def __init__(self, args):
        self.mode = args.mode
        self.device = args.device
        self.saveDir = args.saveDir

        # -------------------- Envrionment configuration -------------------- #
        self.nAgents = args.nAgents
        self.obsDim = args.obsDim

        # -------------------- model configuration -------------------- #
        self.version = args.version
        self.hiddenDim = args.hiddenDim
        self.hiddenCh = args.hiddenCh
        self.outDim = self.nAgents
        self.initSteps = args.initSteps
        self.timeWindow = args.timeWindow
        self.maxPool = args.maxPool
        self.normDotProd = args.normDotProd
        self.stacked = True if self.version=='V2' else False

        self.selectAdversaryModel()

        # -------------------- dataset -------------------- #
        self.dataDir = args.dataDir
        self.valFrac = args.valFrac
        self.batchSize = args.batchSize
        
        # train/val
        trainPath = os.path.join(self.dataDir, 'train_dataset')
        trajDataset = TrajDataset(trainPath, self.nAgents, self.obsDim, self.initSteps, stacked=self.stacked)
        datasetSize = len(trajDataset)
        valSize = int(self.valFrac*datasetSize)
        trainSize = datasetSize - valSize
        self.trainDataset, self.valDataset = random_split(trajDataset, [trainSize, valSize])
        self.trainLoader = DataLoader(self.trainDataset, batch_size=self.batchSize, shuffle=True, num_workers=10)
        self.valLoader = DataLoader(self.valDataset, batch_size=self.batchSize, shuffle=False, num_workers=10)
        self.saveInterval = args.saveInterval

        # test
        testPath = os.path.join(self.dataDir, 'test_dataset')
        self.testDataset = TrajDataset(root_dir=testPath, num_agents=self.nAgents, obs_dim=self.obsDim, init_steps=self.initSteps, stacked=self.stacked)
        self.testLoader = DataLoader(self.testDataset, batch_size=self.batchSize, shuffle=True, num_workers=10)
        self.loadRun = args.loadRun
        self.loadCkpt = args.loadCkpt
        self.visualize = args.visualize

        # -------------------- trainer configuration -------------------- #
        self.criterion = nn.CrossEntropyLoss()
        self.lr = args.lr
        self.lossFunc = lossFunc
        if args.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.adversaryModel.parameters(), lr=self.lr,
                                momentum=0.9, weight_decay=3e-4)
        else:
            self.optimizer = torch.optim.Adam(self.adversaryModel.parameters(), lr=self.lr)
        self.schedulerLam = args.schedulerLam
        self.schedulerSteps = args.schedulerSteps
        lr_func = lambda e: self.schedulerLam**e
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_func)
        self.epochs = args.epochs
        
    def selectAdversaryModel(self):
        if self.version == 'V0':    # LSTM
            inpDim = self.nAgents*self.obsDim
            self.adversaryModel = adversaryNetV0(inpDim=inpDim, hiddenDim=self.hiddenDim, outDim=self.outDim, initSteps=self.initSteps).to(self.device)
        elif self.version == 'V1':  # Scalable LSTM
            self.adversaryModel = adversaryNetV1(inpDim=self.obsDim, hiddenDim=self.hiddenDim, initSteps=self.initSteps).to(self.device)
        elif self.version == 'V2':  # Amanda
            inpDims = (self.nAgents, self.obsDim, self.timeWindow)
            self.adversaryModel = adversaryNetV2(inpDims=inpDims, hiddenCh=self.hiddenCh, hiddenDim=self.hiddenDim, outDim=self.nAgents, applyMaxPool=self.maxPool).to(self.device)
        elif self.version == 'V3':  # Scalable LSTM wt self attention
            self.adversaryModel = adversaryNetV3(inpDim=self.obsDim, hiddenDim=self.hiddenDim, initSteps=self.initSteps, normDotProd=self.normDotProd).to(self.device)
        else:
            print('Please pass valid model version')
            sys.exit()

    def train(self, args):
        # -------------------- walk through exisiting directories -------------------- #
        rootDir = self.saveDir
        if not os.path.exists(rootDir):
            os.makedirs(rootDir)
            runID = 0
        else:
            runs = [int(d.split('_')[-1]) for d in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir,d))]
            runID = max(runs)+1 if len(runs)>0 else 0


        # -------------------- save config and tensorboard -------------------- #
        path = os.path.join(rootDir,'run_{}'.format(runID))
        os.makedirs(path)
        config = vars(args)
        config['runID'] = runID 
        f = open(os.path.join(path,'config.txt'),'w')
        f.write(str(config))
        f.close()
        # -------------------- save config to common file -------------------- #
        f = open(os.path.join(rootDir,'all_config.txt'),'a')
        f.write('\n\n'+str(config))
        f.close()
        
        self.writer = SummaryWriter(path)
        pytorch_total_params = sum(p.numel() for p in self.adversaryModel.parameters())
        print("No. of parameters", pytorch_total_params)
        print('Starting Training')
        tStart = time.time()

        for e in range(self.epochs):
            lossEpoch, valLossEpoch, c = 0, 0, 0
            for X,Y in self.trainLoader:
                self.optimizer.zero_grad()
                outputs = self.adversaryModel(X.to(self.device))
                loss = self.lossFunc(outputs.to(self.device), Y.to(self.device), self.criterion)
                loss.backward()
                self.optimizer.step()
                lossEpoch += loss.data
                c += 1
            lossEpoch /= c
            self.writer.add_scalar('training loss', lossEpoch, e)

            
            c = 0
            for X,Y in self.valLoader:
                with torch.no_grad():
                    outputs = self.adversaryModel(X.to(self.device))
                    loss = self.lossFunc(outputs.to(self.device), Y.to(self.device), self.criterion)
                    valLossEpoch += loss.data
                    c += 1
            valLossEpoch /= c
            self.writer.add_scalar('validation loss', valLossEpoch, e)
            print('Epoch {}, Train loss {}, Val loss {}'.format(e, lossEpoch, valLossEpoch))
            
            if e%self.schedulerSteps == 0:
                self.scheduler.step()

            if e%self.saveInterval == self.saveInterval-1:
                torch.save(self.adversaryModel.state_dict(), os.path.join(path, 'ckpt_{}.pt'.format(e)))



        tEnd = time.time()
        print('Training finished in {} seconds'.format(tEnd-tStart))

    def test(self):
        rootDir = self.saveDir
        print('Loading Run {}, Checkpoint {}'.format(self.loadRun, self.loadCkpt))
        path = os.path.join(rootDir, 'run_'+str(self.loadRun), 'ckpt_{}.pt'.format(self.loadCkpt)) 
        self.adversaryModel.load_state_dict(torch.load(path))

        pytorch_total_params = sum(p.numel() for p in self.adversaryModel.parameters())
        print("No. of parameters", pytorch_total_params)
        # tensor_list = list(self.adversaryModel.items())
        # for layer_tensor_name, tensor in tensor_list:
        #     print('Layer {}: {} elements'.format(layer_tensor_name, torch.numel(tensor)))

        self.adversaryModel.eval()

        print('Starting Testing')
        tStart = time.time()
        loss, testLoss, c = 0, 0, 0
        for X,Y in self.testLoader:
            with torch.no_grad():
                outputs = self.adversaryModel(X.to(self.device))
                loss = self.lossFunc(outputs.to(self.device), Y.to(self.device), self.criterion)
                testLoss += loss.data
                c += 1
        testLoss /= c
        print('Test loss {}'.format(testLoss))
        
        tEnd = time.time()
        print('Testing finished in {} seconds'.format(tEnd-tStart))

        # -------------------- visualize -------------------- #
        if self.visualize:
            figDir = os.path.join(rootDir, 'run_'+str(self.loadRun), 'figures')
            paths = [os.path.join(figDir, path) for path in ['uniform_viz', 'leader_viz']]
            for path in paths:
                if not os.path.exists(path):
                    os.makedirs(path)
            
            print(len(self.testDataset))
            accuracy_dict = {'last_step_correct': []}
            vizLoader = DataLoader(self.testDataset, batch_size=1, shuffle=False)
            for i, (traj, leader_id) in enumerate(vizLoader):
                num_steps = traj.shape[1]
                with torch.no_grad():
                    out = self.adversaryModel(traj.to(self.device))
                    loss = self.lossFunc(out.to(self.device), leader_id.to(self.device), self.criterion)
                    print(loss.data)
                    out = out.detach().cpu().numpy().reshape(-1,self.nAgents)
                    out = np.argmax(out, axis = 1)
                leader_id = leader_id[0].numpy()
                # num_pred_steps = out.shape[1]
                print(traj.shape, out.shape)
                plot_trajectories(traj.numpy().reshape(num_steps, self.nAgents, self.obsDim), leader_id, pred = out, initSteps = self.initSteps,  leader_viz = False, fname = os.path.join(paths[0],'uniform_viz_{}'.format(i)))
                # plot_trajectories(traj.numpy().reshape(num_steps, self.nAgents, self.obsDim), leader_id, pred = out, initSteps = self.initSteps, leader_viz = True, fname = os.path.join(paths[1],'leader_viz_{}'.format(i)))
                
                # save whether final step prediction was correct 
                # [num_steps-init_steps,num_agentsents]
                final_pred = out[-1]
                accuracy_dict['last_step_correct'].append(final_pred == leader_id)
            pd.DataFrame(accuracy_dict).to_csv(os.path.join(rootDir, 'run_{}'.format(self.loadRun),'last_step_prediction_accuracy.csv'))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Arguments for adversary training')
    # -------------------- environment configuration -------------------- #
    parser.add_argument('--nAgents', type=int, default=6, help='Number of agents in swarm including leader')
    parser.add_argument('--obsDim', type=int, default=2, help='Observation dimension corresponding to each agent')
    # -------------------- model configuration -------------------- #
    parser.add_argument('--version', type=str, default='V1', help='Version of adversary model')
    parser.add_argument('--hiddenDim', type=int, default=12, help='Hidden dimension of LSTM or NN')
    parser.add_argument('--hiddenCh', type=int, default=16, help='Channels in hiddden conv layer')
    parser.add_argument('--initSteps', type=int, default=1, help='Number of steps for initializing LSTM hidden states')
    parser.add_argument('--timeWindow', type=int, default=51, help='Number of time steps of observation, used only for Adversary version V2')
    parser.add_argument('--maxPool', default=True, action='store_true', help='Whether to use maxpool after conv layer')
    parser.add_argument('--normDotProd', default=False, action='store_true', help='Whether to normalize dot product, currently only used with version 3')
    # -------------------- training configuration -------------------- #
    parser.add_argument('--device', type=str, default='cuda', help='Choose between {cpu or cuda}')
    parser.add_argument('--batchSize', type=int, default=20, help='Batch size')
    parser.add_argument('--valFrac', type=float, default=0.2, help='Fraction of training data to be used for validation')
    parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate')
    parser.add_argument('--schedulerLam', type=float, default=0.99, help='Decay factor for learning rate scheduler')
    parser.add_argument('--schedulerSteps', type=int, default=1, help='Steps after which lr is decayed')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    # -------------------- data, saveDir and checkpoint -------------------- #
    parser.add_argument('--dataDir', type=str, default='../trajectory_datasets/dataset_1', help='Path to dataset')
    parser.add_argument('--saveDir', type=str, default='./runs', help='Path save model')
    parser.add_argument('--saveInterval', type=int, default=10, help='Save interval')
    parser.add_argument('--loadRun', type=int, default=4, help='Experiment run number')
    parser.add_argument('--loadCkpt', type=int, default=199, help='Model checkpoint to load')
    # -------------------- train/val mode --------------------#
    parser.add_argument('--mode', type=str, default='train', help='Choose between {train, test}')
    parser.add_argument('--visualize', action='store_true', help='Save trajectory plots while testing. And save predictions at final step in csv for plotting accuracy curves')
    # -------------------- parse all the arguments -------------------- #
    args = parser.parse_args()
    # -------------------- train/test -------------------- #
    adversary = AdversaryClass(args)
    if args.mode == 'train':
        adversary.train(args)
    elif args.mode == 'test':
        adversary.test()


