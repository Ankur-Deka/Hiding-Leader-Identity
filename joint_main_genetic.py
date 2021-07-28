import os, sys, shutil
sys.path.append('./mape')
# sys.path.append('./swarm_training')
# sys.path.append('./adversary_training')
sys.path.append('./SIGS-Grid-Search')

import json
import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from swarm_training import utils as swarm_utils
import random
from copy import deepcopy
from swarm_training.utils import normalize_obs, extract_data
from joint_arguments import get_args
from tensorboardX import SummaryWriter
# from swarm_training.eval import evaluate
from swarm_training.learner import setup_master

from adversary_training.models import *    # adversary nets
from adversary_training.dataset import TrajDataset
from adversary_training.utils import lossFunc as adversary_loss_func, plot_trajectories
from joint_utils import obsBuffer

from grid_search import insert_to_csv
from pprint import pprint
import time
import pygmo as pg

# np.set_printoptions(suppress=True, precision=4)
class flockingProblem():
    def __init__(self, args, env, obs_buffer, swarmMaster, adversaryModel, use_adversary, adversary_criterion,
                shuffleLeaderID, sortLeaderID, stackData):
        self.args = args
        self.env = env  # pseudoVecEnv
        self.obs_buffer = obs_buffer
        self.swarmMaster = swarmMaster
        self.adversaryModel = adversaryModel
        self.use_adversary = use_adversary
        self.adversary_criterion = adversary_criterion
        self.shuffleLeaderID = shuffleLeaderID
        self.sortLeaderID = sortLeaderID
        self.stackData = stackData

    def get_bounds(self):
        lb = [0.5, 0, 0, 0, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1]
        ub = [1.5, 1, 1, 2,   2,   2,   2, 1.5, 1.5, 1.5,   2,   2,   2]
        return lb, ub

    

    #   def unstack_data(self, trajs, leaderIDs, predIDs):
    #   # input shapes
    #   # [b, n_agents, obs_dim, steps] & [b] & [b]
    #   b, _, _, steps = trajs.shape
    #   trajs = trajs.transpose((0,3,2,1))  # [b, steps, obs_dim, n_agents]
    #   trajs = trajs.reshape((b, steps, -1)) # [b, steps, obs_dim*n_agents]

    #   leaderIDs = leaderIDs.repeat(steps - self.args.adver)



    # run one episode and return mean of reward
    def fitness(self, params):
        obs = self.env.reset() # although it also auto resets
        # self.eval_obs_buffer.addObs(obs)
        # cur_leader_name = self.eval_env.world.leader_name
            # leader_names.append({'episode':t, 'leaderName':cur_leader_name, 'algo_stage':self.args.algo_stage})

        episode_task_rewards = np.zeros(self.env.n)
        for i in range(10):
            done = [False]*self.env.n
            episode_steps = 0
            print(i)
            while not np.any(done):
                self.obs_buffer.addObs(obs)
                with torch.no_grad():
                    actions = self.swarmMaster.eval_act(obs,params)
                self.env.render()
                obs, reward, done, info = self.env.step(actions)
                # print('obs', obs)
                episode_task_rewards += np.array(reward)[0]

                # -------------------- add terminal state -------------------- #
                if i==0 and episode_steps == args.num_steps_episode-1:
                    obs_terminal = np.array([env_info['terminal_observation'] for env_info in info])
                    self.obs_buffer.addObs(obs_terminal)

                episode_steps+=1


        avg_per_step_task_reward = (episode_task_rewards/episode_steps).mean()
        avg_per_step_task_obj = (1-avg_per_step_task_reward/3)    # make "good flocking to be roughly within performance of 1", like in original repo
        
        # -------------------- privacy reward from adversary -------------------- #
        if self.args.use_adversary:
            trajs, leaderIDs = self.obs_buffer.getData()
            trajs, leaderIDs = self.shuffleLeaderID(trajs, leaderIDs)
            trajs, leaderIDs = self.stackData(trajs, leaderIDs)

            with torch.no_grad():
                outputs = self.adversaryModel(torch.tensor(trajs, dtype=torch.float32).to(self.args.device))
                # print('outputs', outputs)
                # predIDs = torch.argmax(outputs, dim = -1).to('cpu').numpy()
                adv_loss = adversary_loss_func(outputs.to(self.args.device), torch.LongTensor(leaderIDs).to(self.args.device), self.adversary_criterion)
                # print('true, pred', leaderIDs, predIDs)
            # trajs, leaderIDs, predIDs = self.unstack_data(trajs, leaderIDs, predIDs)
            # trajs, leaderIDs, predIDs = self.sortLeaderID(trajs, leaderIDs, predIDs)

            # compute cross entropy loss for privacy
            privacy_obj = 1/(adv_loss.to('cpu').item()+1)

            # privacy_reward = self.args.privacy_reward*(1-(leaderIDs==predIDs).mean())
            # privacy_rewards = self.compPrivacyReward(leaderIDs, predIDs)    # [num_processes, num_steps, num_agents]
            # avg_per_step_privacy_reward = privacy_rewards.mean()    
        
        # -------- use only decent trajs for training adversary -------- #
        lam = 0.8    
        self.obs_buffer.dumpTrajs(save=(avg_per_step_task_reward>lam))
        # print('task and privacy, joint_main_genetic', avg_per_step_task_reward, privacy_reward)
        # Consider privacy only if flocking is decent, Eq. 23 in Paper
        K, beta = 1, self.args.adversary_genetic_beta
        if avg_per_step_task_reward > K:
            total_obj = avg_per_step_task_obj
        else:
            total_obj = beta*avg_per_step_task_obj + (1-beta)*privacy_obj

        return [total_obj]


class jointController():
    def __init__(self, args):
        self.args = args
        if self.args.mode == 'train':
            self.swarmMaster = setup_master(self.args)
        self.initializeAdversary()
        max_files = 100
        self.update_counter = 0
        self.save_counter = 0
        stacked = self.args.adversary_version == 'V2'
        if self.args.mode == 'test':
            self.eval_swarm_master, self.eval_env = setup_master(self.args, return_env=True, goal_at_top=args.goal_at_top)
        if self.args.mode == 'train':
            self.obs_buffer = obsBuffer(args.num_agents, args.obs_dim, args.num_processes, args.num_steps_episode, args.adversary_init_steps, max_files, args.data_temp_dir)  # stores data to train adversary
        if self.args.mode == 'test':
            self.eval_obs_buffer = obsBuffer(args.num_agents, args.obs_dim, 1, args.num_steps_episode, args.adversary_init_steps, max_files, os.path.join(args.out_dir, 'trajs'))
        
        self.adversary_criterion = nn.CrossEntropyLoss()
        # self.adversary_optimizer = torch.optim.Adam(self.adversaryModel.parameters(), lr=self.args.adversary_lr)
        self.adversary_optimizer = torch.optim.SGD(self.adversaryModel.parameters(), lr=1e-3,
                                momentum=0.9, weight_decay=3e-4)

    def initializeAdversary(self):
        if self.args.adversary_version == 'V0':
            inpDim = self.args.num_agents*self.args.obs_dim
            self.adversaryModel = adversaryNetV0(inpDim=inpDim, hiddenDim=self.hiddenDim, outDim=self.outDim, initSteps=self.initSteps).to(self.device)
        elif self.args.adversary_version == 'V1':
            self.adversaryModel = adversaryNetV1(inpDim=self.args.obs_dim, hiddenDim=self.args.adversary_hidden_dim, initSteps=1).to(self.args.device)
        elif self.args.adversary_version == 'V2':
            inpDims = (self.args.num_agents, self.args.obs_dim, self.args.num_steps_episode+1)
            self.adversaryModel = adversaryNetV2(inpDims=inpDims, hiddenCh=self.args.adversary_hidden_ch, hiddenDim=self.args.adversary_hidden_dim, outDim=self.args.num_agents, applyMaxPool=self.args.adversary_apply_maxpool).to(self.args.device)
        elif self.args.adversary_version == 'V3':
            self.adversaryModel = adversaryNetV3(inpDim=self.obsDim, hiddenDim=self.hiddenDim, initSteps=self.initSteps, normDotProd=self.normDotProd).to(self.device)
        else:
            print('Please pass valid model version')
            sys.exit()


    def loadPretrained(self):
        self.loadPretrainedSwarm()
        self.loadPretrainedAdversary()
        
        print('Loaded pretrained swarm and adversary models')

    def loadPretrainedSwarm(self):
        # -------- load swarm model -------- #
        checkpoint = torch.load(args.swarm_load_path, map_location=lambda storage, loc: storage)
        policies_list = checkpoint['models']
        self.swarmMaster.load_models(policies_list)
    
    def loadPretrainedAdversary(self):
        # --------load adversary model -------- # 
        self.adversaryModel.load_state_dict(torch.load(self.args.adversary_load_path))


    # shuffles the leader ID
    def shuffleLeaderID(self, trajs, leaderIDs, randIDs=None):
        # trajs: [num_processes, num_steps, num_agents*obs_dim]
        # leaderIDs: [num_processes, num_steps]
        if randIDs is None:
            randIDs = np.random.randint(0, self.args.num_agents, self.args.num_processes)
        for X, ID, Y in zip(trajs, randIDs, leaderIDs):
            temp = X[:,ID*self.args.obs_dim:(ID+1)*self.args.obs_dim].copy()
            X[:,ID*self.args.obs_dim:(ID+1)*self.args.obs_dim] = X[:,0:self.args.obs_dim].copy()
            X[:,0:self.args.obs_dim] = temp.copy()

            Y[:] = ID

        return trajs, leaderIDs

    # makes the first ID the leader ID
    def sortLeaderID(self, trajs, trueLeaderIDs, predLeaderIDs):
        # leaderIDs: [num_processes, num_steps-init_steps]
        for traj, ID, predID in  zip(trajs, trueLeaderIDs, predLeaderIDs):
            i = int(ID[0])
            temp = traj[:,i*self.args.obs_dim:(i+1)*self.args.obs_dim].copy()
            traj[:,i*self.args.obs_dim:(i+1)*self.args.obs_dim] = traj[:,0:self.args.obs_dim]
            traj[:,0:self.args.obs_dim] = temp
            
            pos1 = np.where(predID==0)
            pos2 = np.where(predID==ID)
            predID[pos1] = ID[0]
            predID[pos2] = 0
            ID[:] = 0
        return trajs, trueLeaderIDs, predLeaderIDs

    def stackData(self, trajs, leaderIDs):
        # input shapes
        # [batch_size, num_steps, num_agents*obs_dim] & [batch_size, num_steps-init_steps]
        b, steps, _ = trajs.shape
        trajs = trajs.reshape((b, steps, self.args.num_agents, self.args.obs_dim)) # [b, steps, n_agents, obs_dim]


        leaderIDs = leaderIDs[:, 0] # [b]

        return trajs, leaderIDs

    def unstackData(self, trajs, leaderIDs, predIDs):
        # trajs: [b, steps, n_agents, obs_dim]
        # leaderIDs: [b]
        # predIDs: [b]
        
        # trajs -> [b,steps,n_agents*obs_dim]
        b, steps, _, _ = trajs.shape
        trajs = trajs.reshape((b,steps,-1))
        leaderIDs = leaderIDs.reshape((b,1))
        predIDs = predIDs.reshape((b,1))

        return trajs, leaderIDs, predIDs



    def compPrivacyReward(self, leaderIDs, predIDs):
        privacy_rewards = -self.args.privacy_reward*(leaderIDs==predIDs)         # [num_processes, num_steps-init_steps]
        privacy_rewards = np.repeat(privacy_rewards[:,:,np.newaxis], self.args.num_agents, axis = 2)  # [num_processes, num_steps-init_steps, num_agents]
        return privacy_rewards

    def should_update(self, i):
        num_frames = (i+1)*self.args.num_processes*self.args.num_steps_episode                   # num frames
        decision = num_frames//self.args.update_every > self.update_counter 
        if decision:
            self.update_counter+=1
        return decision

    def should_save(self, i):
        num_frames = (i+1)*self.args.population_size*self.args.num_steps_episode                   # num frames
        decision = num_frames//args.save_interval > self.save_counter 
        if decision:
            self.save_counter+=1
        return decision

    def reset_counters(self):
        self.update_counter = 0
        self.save_counter = 0
        if self.args.continue_training:
            self.save_counter = self.args.load_ckpt
            self.update_counter = self.args.load_ckpt * self.args.save_interval // self.args.update_every
            
    def saveLeaderPreds(self, episode, leaderIDs, predIDs, probs):
        # header
        header = 'leaderID,predID'
        for i in range(self.args.num_agents):
            header+=',{}'.format(i)

        # save to file
        # print('shapes, joint main genetic')
        # print(leaderIDs.shape, predIDs.shape, probs.shape)
        data = np.concatenate((leaderIDs, predIDs, np.squeeze(probs, axis=0))).reshape(1,-1)
        np.savetxt(os.path.join(self.args.out_dir, 'adversary_preds', str(episode)+'.csv'), data, delimiter=',', header=header)

        # # plot probs assigned to true leader
        # probs_true_id = probs[0,int(leaderIDs[0,0])]
        # sns.set(style='darkgrid')
        # graph = sns.lineplot(np.arange(0,self.args.num_steps_episode), probs_true_id)
        # graph.axhline(0,xmin=0.05, xmax=0.95, linestyle='--')
        # plt.xlabel('Time step')
        # plt.ylabel('Prob. assigned to true leader')
        # plt.ylim(-0.05,1.05)
        # plt.tight_layout()
        # plt.savefig(os.path.join(self.args.out_dir, 'adversary_preds', str(episode)+'.png'), dpi=100)
        # plt.clf()

    # def get_params(self):
    #     # v_leader, leader_sep_w, leader_ali_w, leader_coh_w, leader_sep_max_cutoff, leader_ali_r, leader_coh_r
    #     # sep_w, ali_w, coh_w, sep_max_cutoff, ali_r, coh_r
    #     return [1, 0.5, 0.5, 0.5, 2, 2, 2, 0.5, 0.5, 0.5, 2, 2, 2]


    def get_bounds(self):
        lb = [0.5, 0, 0, 0, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1]
        ub = [1.5, 1, 1, 2,   2,   2,   2, 1.5, 1.5, 1.5,   2,   2,   2]
        return lb, ub
    
    def get_random_params(self):
        lb, up = self.get_bounds()
        params = np.random.uniform(lb, up)
        return params

    def evaluate(self, seed, swarm_checkpoint=None, adversary_checkpoint=None):
        self.eval_env.seed(seed)
        if self.args.algo == 'genetic':
            swarm_params = swarm_checkpoint['champion_params'].numpy()
            print('loaded swarm')
            print(swarm_params)
            print(swarm_checkpoint['all_params'])
        if self.args.use_adversary:
            # print(self.args.load_adversary, type(self.args.lo))
            self.adversaryModel.load_state_dict(adversary_checkpoint)
        
        num_eval_episodes = self.args.num_eval_episodes
        episode_task_rewards = np.full((num_eval_episodes, self.eval_env.n), 0.0)
        per_step_task_rewards = np.full((num_eval_episodes, self.eval_env.n), 0.0)
        episode_privacy_rewards = np.full((num_eval_episodes, self.eval_env.n), 0.0)
        
        # world.dists at the end of episode for simple_spread
        final_min_dists = []
        leader_names = []
        num_success = 0
        episode_length = 0

        save = not (self.args.out_dir is None)
        if save:
            paths = [os.path.join(self.args.out_dir, 'traj_plots', p) for p in ['uniform_viz', 'leader_viz']]
            paths.append(os.path.join(self.args.out_dir, 'adversary_preds'))
            for p in paths:
                if not os.path.exists(p):
                    os.makedirs(p)

        obs = self.eval_env.reset() # although it also auto resets
        for t in range(num_eval_episodes):
            # print('obs', obs)

            self.eval_obs_buffer.addObs(obs)
            cur_leader_name = self.eval_env.world.leader_name
            leader_names.append({'episode':t, 'leaderName':cur_leader_name, 'algo_stage':self.args.algo_stage})

            # -------- recording video -------- #
            if self.args.record_video:
                video_name = 'same_color_{}_{}_{}_{}.{}'.format(self.args.same_color, self.args.load_run,self.args.load_ckpt, t, self.args.video_format) if self.args.store_video_together else '{}.{}'.format(t, self.args.video_format)
                video_path = os.path.join(self.args.video_path, video_name)
                print(video_path)
                self.eval_env.startRecording(video_path)

        
            done = [False]*self.eval_env.n
            episode_steps = 0
            if self.args.render:
                attn = None# if not render_attn else self.eval_swarm_master.team_attn
                if attn is not None and len(attn.shape)==3:
                    attn = attn.max(0)
                self.eval_env.render(attn=attn)
                
            if self.args.algo=='genetic_random':
                swarm_params = self.get_random_params()

            while not np.any(done):
                actions = []
                with torch.no_grad():
                    actions = self.eval_swarm_master.eval_act(obs,swarm_params)
            
                episode_steps += 1
                obs, reward, done, info = self.eval_env.step(actions)
                # print('obs', obs)
                episode_task_rewards[t] += np.array(reward)[0]
                
                if np.any(done):
                    obs_terminal = np.array([env_info['terminal_observation'] for env_info in info])
                    self.eval_obs_buffer.addObs(obs_terminal)
                else:   # vec_env auto_resets
                    self.eval_obs_buffer.addObs(obs)

                if self.args.render:
                    # time.sleep(0.1)
                    attn = None# if not render_attn else self.eval_swarm_master.team_attn
                    if attn is not None and len(attn.shape)==3:
                        attn = attn.max(0)
                    self.eval_env.render(attn=attn)

                if self.args.record_video:
                    self.eval_env.recordFrame()

                path = 'output/Frames/Zheng_{}_{}.png'.format(t,episode_steps)
                self.eval_env.saveFrame(path)
                #     time.sleep(0.08)

            # -------------------- privacy reward from adversary -------------------- #
            if self.args.use_adversary:
                # testPath = 'trajectory_datasets/dataset_1/test_dataset'
                # testDataset = TrajDataset(root_dir=testPath, num_agents=self.args.num_agents, obs_dim=self.args.obs_dim, init_steps=self.args.adversary_init_steps)
                # trajs, leaderIDs = testDataset[0]
                # trajs, leaderIDs = trajs.view(1,51,12), leaderIDs.view(1,50)
                trajs, leaderIDs = self.eval_obs_buffer.getData()
                
                randIDs = np.array([cur_leader_name]) if self.args.random_leader_name else None
                # print(self.args.random_leader_name, randIDs)
                trajs, leaderIDs = self.shuffleLeaderID(trajs, leaderIDs, randIDs)
                trajs, leaderIDs = self.stackData(trajs, leaderIDs)
                with torch.no_grad():
                    outputs = self.adversaryModel(torch.tensor(trajs, dtype=torch.float32).to(self.args.device))
                    predIDs = torch.argmax(outputs, dim = -1).cpu().numpy().astype(int)
                    # print('predIDs', predIDs)
                    # print('leaderIDs', leaderIDs)
                    # loss = adversary_loss_func(outputs.to(self.args.device), torch.tensor(leaderIDs, dtype = torch.int64).to(self.args.device), self.adversary_criterion)
                    # print(predIDs)
                    # print(leaderIDs)
                    # print(leaderIDs)
                    # print(predIDs)
                    # print(privacy_reward)
                    # ID = int(leaderIDs[0,0])
                  
                    # print('softmax values', torch.softmax(outputs, dim = -1)[0,:,ID])
                    # print(loss.data)

                
                if save:
                    probs = torch.softmax(outputs, dim = -1).cpu().numpy()
                    self.saveLeaderPreds(t, leaderIDs, predIDs, probs)

                trajs, leaderIDs, predIDs = self.unstackData(trajs, leaderIDs, predIDs)
                trajs, leaderIDs, predIDs = self.sortLeaderID(trajs, leaderIDs, predIDs)
                privacy_reward = self.args.privacy_reward*(1-(leaderIDs==predIDs).mean())
                
                # privacy_rewards = self.compPrivacyReward(leaderIDs, predIDs)[0].sum(axis = 0)    # [num_processes=1, num_steps, num_agents] -> [num_agents]
                # print(privacy_rewards)
                episode_privacy_rewards[t] = privacy_reward

                # trajs, leaderIDs = self.obs_buffer.getData()
                # trajs, leaderIDs = self.shuffleLeaderID(trajs, leaderIDs)

                # with torch.no_grad():
                #     outputs = self.adversaryModel(torch.tensor(trajs, dtype=torch.float32).to(self.args.device))
                #     predIDs = torch.argmax(outputs, dim = -1).to('cpu').numpy()
                
                # trajs, leaderIDs, predIDs = self.unstack_data(trajs, leaderIDs, predIDs)
                # trajs, leaderIDs, predIDs = self.sortLeaderID(trajs, leaderIDs, predIDs)
                # privacy_reward = self.args.privacy_reward*(1-(leaderIDs==predIDs).mean())
            # -------------------- -------------------- -------------------- #
            self.eval_obs_buffer.dumpTrajs(counter = t, save = save)

            # -------------------- trajectory plots --------------------#
            if save and self.args.plot_trajectories and self.args.use_adversary:
                plot_trajectories(trajs[0], leaderIDs[0][0], initSteps = self.args.num_steps_episode, pred = predIDs[0], leader_viz = False, fname = os.path.join(self.args.out_dir, 'traj_plots/uniform_viz','uniform_viz_{}'.format(t)))
                plot_trajectories(trajs[0], leaderIDs[0][0], initSteps = self.args.num_steps_episode, pred = predIDs[0], leader_viz = True, fname = os.path.join(self.args.out_dir, 'traj_plots/leader_viz','leader_viz_{}'.format(t)))

            # -------------------- -------------------- -------------------- #
            per_step_task_rewards[t] = episode_task_rewards[t]/episode_steps
            # rew_data = [self.args.load_run]+[t]+list(per_step_rewards[t])+list(episode_privacy_rewards[t])
            # insert_to_csv('output/reward_data.csv', rew_data)

            num_success += info[0]['is_success']
            episode_length = (episode_length*t + episode_steps)/(t+1)

            # for simple spread self.eval_env only
            if self.args.env_name == 'simple_spread':
                final_min_dists.append(self.eval_env.world.min_dists)
            elif self.args.env_name == 'simple_formation' or self.args.env_name=='simple_line':
                final_min_dists.append(self.eval_env.world.dists)

            if self.args.render:
                print("Ep {} | Success: {} \n Av per-step reward: {:.2f} | Ep Length {}".format(t,info[0]['n'][0]['is_success'],
                    per_step_task_rewards[t][0],info[0]['n'][0]['world_steps']))
            

            if self.args.record_video:
                self.eval_env.endVideo()

        reward_dict = {}
        reward_dict['Task reward'] = per_step_task_rewards.flatten().tolist()
        reward_dict['Privacy reward'] = (episode_privacy_rewards).flatten().tolist()
        reward_dict['Algorithm'] = ['Zheng et al.']*(self.args.num_eval_episodes*self.args.num_agents)
        pd.DataFrame(reward_dict).to_csv('rewards.csv', mode='a', header=False)

        if self.args.record_video:
            with open(os.path.join(self.args.out_dir, 'leader_names_in_video.txt'), 'w') as f:
                f.write(str(leader_names))


        # print(locals().keys())
        # print('ankur', locals()['episode_length'], locals()['all_episode_rewards'])
        return {'episode_task_rewards': episode_task_rewards, 'per_step_task_rewards': per_step_task_rewards, 'episode_privacy_rewards': episode_privacy_rewards, 'final_min_dists': final_min_dists, 'num_success': num_success, 'episode_length': episode_length}
         


    def train(self, swarm_checkpoint, adversary_checkpoint, return_early=False):
        if not swarm_checkpoint is None:
            policies_list = swarm_checkpoint['models']
            self.swarmMaster.load_models(policies_list)
        
        if not adversary_checkpoint is None:
            self.adversaryModel.load_state_dict(adversary_checkpoint)

        self.reset_counters()
        writer = SummaryWriter(self.args.log_dir)    
        envs = swarm_utils.make_parallel_envs(self.args) 

        prob = flockingProblem(args, envs, self.obs_buffer, self.swarmMaster, self.adversaryModel, self.args.use_adversary, self.adversary_criterion, self.shuffleLeaderID, self.sortLeaderID, self.stackData)

        pop = pg.population(prob, size=self.args.population_size)
        algo = pg.algorithm(pg.sga(gen = 1, cr = .90, m = 0.02, param_s = 3, crossover = "single", mutation = "uniform", selection = "truncated"))
        # -------------------- holding data -------------------- #
        # episode_rewards = torch.zeros([self.args.num_processes, self.args.num_agents], device=args.device)
        # final_rewards = torch.zeros([self.args.num_processes, self.args.num_agents], device=self.args.device)
        
        # used during evaluation only
        # eval_master, eval_env = setup_master(args, return_env=True) 
        # obs = envs.reset() # shape - num_processes x num_agents x obs_dim
        
        # # start simulations
        # start = datetime.datetime.now()

        for i in range(self.args.continue_from_iter, self.args.continue_from_iter + self.args.num_train_iters):
            # t_start = time.time()
            # -------- run one paralle episode for each process -------- #
            pop = algo.evolve(pop)
            champion_rew = -pop.champion_f
            print('champion_rew', champion_rew)
            num_env_steps = (i+1)*self.args.population_size*self.args.num_steps_episode
            writer.add_scalar('champion_obj', champion_rew, num_env_steps)

            # add info of all individuals
            all_rews = -pop.get_f() # we -(-reward)
            for j,rew in enumerate(all_rews):
                writer.add_scalar('Genetic_obj/individual_{}'.format(j), rew, num_env_steps)

            # -------------------- adversary training -------------------- #
            if self.args.train_adversary:
                print('Training adversary')
                trajDataset = TrajDataset(self.args.data_temp_dir, self.args.num_agents, self.args.obs_dim, self.args.adversary_init_steps, stacked=True)
                trajLoader = DataLoader(trajDataset, batch_size=self.args.num_processes, shuffle=True, num_workers=10)
                # trajLoader = DataLoader(trajDataset, batch_size=1, shuffle=True, num_workers=1)
                avg_loss = 0
                for e in range(self.args.adversary_num_epochs):
                    lossEpoch, valLossEpoch, c = 0, 0, 0
                    for X,Y in trajLoader:
                        self.adversary_optimizer.zero_grad()
                        outputs = self.adversaryModel(X.to(self.args.device))
                        loss = adversary_loss_func(outputs.to(self.args.device), Y.to(self.args.device), self.adversary_criterion)
                        loss.backward()
                        self.adversary_optimizer.step()
                        lossEpoch += loss.data
                        c += 1
                        # print('outputs in training adversary, true, pred, joint-main-genetic')
                        # print(Y, torch.argmax(outputs, dim = -1).to('cpu').numpy())
                        # print('Y')
                        # print(Y)
                        # print('loss')
                    lossEpoch /= c
                    avg_loss += lossEpoch
                avg_loss /= self.args.adversary_num_epochs
                writer.add_scalar('adversary_loss', avg_loss, num_env_steps)
                print('adversary_loss', avg_loss)
            
            if self.should_save(i):
                champion_params = torch.tensor(pop.champion_x)
                all_params = torch.tensor(pop.get_x())
                adversary_state_dict = self.adversaryModel.state_dict() if self.args.use_adversary else None
                savedict = {'swarm': {'champion_params': champion_params, 'all_params': all_params}, 'adversary': adversary_state_dict}
                savedir = args.save_dir+'/ckpt_'+str(self.save_counter)+'.pt'
                torch.save(savedict, savedir)

                # -------- mapping from ckpt to num_frames -------- #
                with open(os.path.join(args.save_dir,'ckpt_to_frames.txt'),'a') as f:
                    f.write(str(self.save_counter) + ' ' + str(num_env_steps)+'\n')

                # -------------------- update observation -------------------- #
                # if args.render and args.num_processes==1:
                #     envs.render()
                    # time.sleep(0.1)
                # print('obs', 'joint_main.py')
                # print(obs)
                # self.swarmMaster.update_obs(obs)
                # self.obs_buffer.addObs(obs)

                # -------------------- get actions and interact -------------------- #
            #     with torch.no_grad():
            #         actions_list = self.swarmMaster.act()
            #     agent_actions = np.transpose(np.array(actions_list),(1,0,2))
            #     t1= time.time()
            #     obs_nxt, reward, done, info = envs.step(agent_actions)
            #     reward = torch.from_numpy(np.stack(reward)).float().to(args.device)
            #     episode_rewards += reward
            #     masks = torch.FloatTensor(1-1.0*done).to(args.device)

              
            #     # -------------------- update rollout. disable auto_terminate, i.e.don't yet compute episode returns, step still gets updated) -------------------- #
            #     if step < args.num_steps_episode-1:
            #         self.swarmMaster.update_obs_nxt(reward, masks, obs_nxt, info)
            #     # vec_env auto_resets. step inside swarm_master & step/start_step in rollout_storage auto updated
            #     else:   
            #         obs_terminal = np.array([env_info['terminal_observation'] for env_info in info])
            #         self.swarmMaster.update_obs_nxt(reward, masks, obs_terminal, info, auto_terminate=False)
            #         self.obs_buffer.addObs(obs_terminal)

            #     obs = deepcopy(obs_nxt)

            # -------------------- privacy reward from adversary -------------------- #
            # if self.args.use_adversary:
            #     trajs, leaderIDs = self.obs_buffer.getData()
            #     trajs, leaderIDs = self.shuffleLeaderID(trajs, leaderIDs)
            #     with torch.no_grad():
            #         outputs = self.adversaryModel(torch.tensor(trajs, dtype=torch.float32).to(self.args.device))
            #         predIDs = torch.argmax(outputs, dim = -1).to('cpu').numpy()
                
            #     trajs, leaderIDs, predIDs = self.sortLeaderID(trajs, leaderIDs, predIDs)
                
            #     privacy_rewards = self.compPrivacyReward(leaderIDs, predIDs)    # [num_processes, num_steps, num_agents]
            #     print('swarm privacy reward', privacy_rewards.mean())
            #     # privacy_rewards[:] = 0 ## watch out
            #     self.swarmMaster.add_to_reward(privacy_rewards)
                
            #     num_env_steps = (i+1)*self.args.num_processes*args.num_steps_episode
            #     for idx in range(self.args.num_agents):
            #         writer.add_scalar('privacy_reward/'+'agent'+str(idx), privacy_rewards[:,:,idx].mean(), num_env_steps)
                



            # print('time taken', time.time()-t_start)
            # # -------------------- wrap up rollouts -------------------- #
            # self.swarmMaster.terminate_episodes()           
            # self.obs_buffer.dumpTrajs()

            # # -------------------- training -------------------- #
            # if self.should_update(i):
            #     # -------------------- adversary training -------------------- #
            #     if self.args.train_adversary:
            #         print('Training adversary')
            #         trajDataset = TrajDataset(self.args.data_temp_dir, self.args.num_agents, self.args.obs_dim, self.args.adversary_init_steps)
            #         trajLoader = DataLoader(trajDataset, batch_size=self.args.num_processes, shuffle=True, num_workers=10)
            #         # trajLoader = DataLoader(trajDataset, batch_size=1, shuffle=True, num_workers=1)
            #         avg_loss = 0
            #         for e in range(self.args.adversary_num_epochs):
            #             lossEpoch, valLossEpoch, c = 0, 0, 0
            #             for X,Y in trajLoader:
            #                 self.adversary_optimizer.zero_grad()
            #                 outputs = self.adversaryModel(X.to(self.args.device))
            #                 loss = adversary_loss_func(outputs.to(self.args.device), Y.to(self.args.device), self.adversary_criterion)
            #                 loss.backward()
            #                 self.adversary_optimizer.step()
            #                 lossEpoch += loss.data
            #                 c += 1
            #                 # print('outputs')
            #                 # print(torch.argmax(outputs, dim = -1).to('cpu').numpy())
            #                 # print('Y')
            #                 # print(Y)
            #                 # print('loss')
            #                 print(loss.data)
            #             lossEpoch /= c
            #             avg_loss += lossEpoch
            #         avg_loss /= self.args.adversary_num_epochs
            #         num_env_steps = (i+1)*args.num_processes*args.num_steps_episode
            #         writer.add_scalar('adversary_loss', avg_loss, num_env_steps)
            #         print('adversary_loss', avg_loss)
            #     # --------------------swarm training-------------------- #
            #     if self.args.train_swarm:
            #         print('Training swarm')
            #         # run update for num_updates times
            #         return_rew = args.algo=='ppo'
            #         return_vals = self.swarmMaster.update(return_rew = return_rew)   # internally considers early termination of all episodes
            #         value_loss = return_vals[:, 0]
            #         action_loss = return_vals[:, 1]
            #         dist_entropy = return_vals[:, 2]

            #         if return_rew:
            #             buffer_avg_reward = return_vals[:, 3]  
            #             print('Buffer avg per step reward {}'.format(buffer_avg_reward))
            #             # tensorboard 
            #             for idx,rew in enumerate(buffer_avg_reward):
            #                 num_env_steps = (i+1)*self.args.num_processes*self.args.num_steps_episode
            #                 writer.add_scalar('train_buffer_avg_per_step_reward/'+'agent'+str(idx), rew, num_env_steps)
                            



        #     total_num_steps = (j + 1) * args.num_processes * args.num_steps

        #     if j%args.log_interval == 0:
        #         end = datetime.datetime.now()
        #         seconds = (end-start).total_seconds()
        #         mean_reward = final_rewards.mean(dim=0).cpu().numpy()
        #         print("Updates {} | Num timesteps {} | Time {} | FPS {}\nMean reward {}\nEntropy {:.4f} Value loss {:.4f} Policy loss {:.4f}\n".
        #               format(j, total_num_steps, str(end-start), int(total_num_steps / seconds), 
        #               mean_reward, dist_entropy[0], value_loss[0], action_loss[0]))
        #         if not args.test:
        #             for idx in range(n):
        #                 writer.add_scalar('agent'+str(idx)+'/training_reward', mean_reward[idx], j)

        #             writer.add_scalar('all/value_loss', value_loss[0], j)
        #             writer.add_scalar('all/action_loss', action_loss[0], j)
        #             writer.add_scalar('all/dist_entropy', dist_entropy[0], j)

        #     if args.eval_interval is not None and j%args.eval_interval==0:
        #         ob_rms = (None, None) if envs.ob_rms is None else (envs.ob_rms[0].mean, envs.ob_rms[0].var)
        #         print('===========================================================================================')
        #         _, eval_perstep_rewards, final_min_dists, num_success, eval_episode_len = evaluate(args, None, master.all_policies,
        #                                                                                            ob_rms=ob_rms, env=eval_env,
        #                                                                                            master=eval_master)
        #         print('Evaluation {:d} | Mean per-step reward {:.2f}'.format(j//args.eval_interval, eval_perstep_rewards.mean()))
        #         print('Num success {:d}/{:d} | Episode Length {:.2f}'.format(num_success, args.num_eval_episodes, eval_episode_len))
        #         if final_min_dists:
        #             print('Final_dists_mean {}'.format(np.stack(final_min_dists).mean(0)))
        #             print('Final_dists_var {}'.format(np.stack(final_min_dists).var(0)))
        #         print('===========================================================================================\n')

        #         if not args.test:
        #             writer.add_scalar('all/eval_success', 100.0*num_success/args.num_eval_episodes, j)
        #             writer.add_scalar('all/episode_length', eval_episode_len, j)
        #             for idx in range(n):
        #                 writer.add_scalar('agent'+str(idx)+'/eval_per_step_reward', eval_perstep_rewards.mean(0)[idx], j)
        #                 if final_min_dists:
        #                     writer.add_scalar('agent'+str(idx)+'/eval_min_dist', np.stack(final_min_dists).mean(0)[idx], j)

        #         curriculum_success_thres = 0.9
        #         if return_early and num_success*1./args.num_eval_episodes > curriculum_success_thres:
        #             savedict = {'models': [agent.actor_critic.state_dict() for agent in master.all_agents]}
        #             ob_rms = (None, None) if envs.ob_rms is None else (envs.ob_rms[0].mean, envs.ob_rms[0].var)
        #             savedict['ob_rms'] = ob_rms
        #             savedir = args.save_dir+'/ep'+str(j)+'.pt'
        #             torch.save(savedict, savedir)
        #             print('===========================================================================================\n')
        #             print('{} agents: training complete. Breaking.\n'.format(args.num_agents))
        #             print('===========================================================================================\n')
        #             break

        # writer.close()
        # if return_early:
        #     return savedir

if __name__ == '__main__':
    args = get_args()
    if args.seed is None:
        args.seed = random.randint(0,10000)
    
    # if args.algo == 'genetic':
    args.num_train_iters = args.num_frames // args.num_steps_episode // args.population_size
    # else:
    #     args.num_train_iters = args.num_frames // args.num_steps_episode // args.num_processes   # no. of training iterations of training loop
    torch.manual_seed(args.seed)
    # torch.set_num_threads(1)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    myController = jointController(args)
    
    pprint(vars(args))

    # -------- loading pretrained models -------- #
    swarm_checkpoint, adversary_checkpoint = None, None
    if not args.load_mode is None:
        if args.load_mode == 'joint':
            print('args.load_path', args.load_path)
            checkpoint = torch.load(args.load_path)
            swarm_checkpoint = checkpoint['swarm']
            adversary_checkpoint = checkpoint['adversary'] if args.use_adversary else None
        elif args.load_mode == 'individual':
            if not args.swarm_load_path is None:
                swarm_checkpoint = torch.load(args.swarm_load_path)
                if 'swarm' in swarm_checkpoint.keys():
                    swarm_checkpoint = swarm_checkpoint['swarm']
            if args.use_adversary:
                adversary_checkpoint = torch.load(args.adversary_load_path)
                print('args.adversary_load_path', args.adversary_load_path)
                if 'adversary' in adversary_checkpoint.keys():
                    adversary_checkpoint = adversary_checkpoint['adversary']
            else:
                adversary_checkpoint = None
            print('loaded swarm and adversary', 'joint_main.py')

    if args.mode == 'train':
        with open(os.path.join(args.save_dir, 'params.json'), 'w') as f:
            params = deepcopy(vars(args))
            params.pop('device')
            json.dump(params, f)
            # also save to a common file
            f = open(os.path.join(args.root_save_dir,'all_config.txt'),'a')
            f.write(str(params)+'\n\n')
            f.close()

        myController.train(swarm_checkpoint, adversary_checkpoint)

    elif args.mode == 'test':
        # Save configs
        if not args.out_dir is None:
            with open(os.path.join(args.out_dir, 'params.json'), 'w') as f:
                params = deepcopy(vars(args))
                params.pop('device')
                json.dump(params, f)
            # save to common file
            with open(os.path.join('output','all_config.txt'),'a') as f:
                f.write('\n\n'+str(params))
        metrics = myController.evaluate(args.seed, swarm_checkpoint, adversary_checkpoint)
        print('Metrics \n{}'.format(metrics))
    