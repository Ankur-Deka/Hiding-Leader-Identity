import os, shutil
import numpy as np

from swarm_training.utils import extract_data_parallel
from copy import deepcopy

class obsBuffer():
    def __init__(self, num_agents, obs_dim, num_processes, num_steps_episode, init_steps, max_count, data_temp_dir):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.num_processes = num_processes
        self.num_steps_episode = num_steps_episode
        self.dir = data_temp_dir
        self.counter = 0    # keeps count of no. of files
        self.init_steps = init_steps
        self.max_count = max_count
        self.buffer = []
        self.reset()

    # fully resets, deletes all previous trajs. run this when you're done with one training iteration (on policy) 
    def reset(self):
        if os.path.exists(self.dir):
            shutil.rmtree(self.dir)
        os.makedirs(self.dir)
        print(self.dir, 'joint_utils.py')
        self.counter = 0
        self.buffer = []

    # add one time step data across parallel envs to the buffer
    def addObs(self, obs):
        # num_process, num_agents, env_obs_dim
        data = extract_data_parallel(obs) # num_processes, num_agents, 2
        self.buffer.append(data)

    # returns the data inside buffer
    def getData(self):
        data = np.array(self.buffer)     # num_steps, num_processes, num_agents*obs_dim+1
        data = np.swapaxes(data, 0, 1)    # num_processes, num_steps, num_agents*obs_dim+1

        trajs, leaderIDs = deepcopy(data[:,:,:-1]), deepcopy(data[:,self.init_steps:,-1]) # [num_processes, num_steps, num_agents*obs_dim] and [num_processes, num_steps-init_steps]
            
        return trajs, leaderIDs

    # save data in buffer as trajectories, do this at the end of an episode
    def dumpTrajs(self, counter = None, save=True):
        if save:
            data = np.array(self.buffer)     # num_steps, num_processes, num_agents*obs_dim+1
            data = np.swapaxes(data, 0, 1)    # num_processes, num_steps, num_agents*obs_dim+1

            for traj in data:
                if counter is None:
                    np.save(os.path.join(self.dir, str(self.counter)),traj)
                    self.counter = (self.counter+1) % self.max_count
                else: 
                    np.save(os.path.join(self.dir, str(counter)),traj)
                    
        self.buffer = []

