import numpy as np
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
import gym, gym_vecenv
import time
# from airsim_envs import gym_airsim

def extract_data(obs):  # extracting position and adding leader label for one time step
    data = np.array(obs)
    data = data[:,2:4].reshape(-1)
    data = np.concatenate((data, [0]))
    return data

def extract_data_parallel(obs): # same as extract_data for parallel envs
    data = np.array(obs)
    num_processes, num_agents, obs_dim = data.shape
    data = data[:, :, 2:4].reshape((num_processes, -1))  # contains position information
    data = np.concatenate((data, np.zeros((num_processes, 1))), axis = 1)
    return data     # [num_process, num_agents*obs_dim+1]

def normalize_obs(obs, mean, std):
    if mean is not None:
        return np.divide((obs - mean), std)
    else:
        return obs

def should_update(j, update_counter, args):
    num_frames = (j+1)*args.num_processes                   # num frames
    return num_frames//args.update_every > update_counter 

def should_eval(j, eval_counter, args):                     # doubles for logging
    num_frames = (j+1)*args.num_processes                   # num frames
    return num_frames//args.eval_interval + 1 > eval_counter 

def should_save(j, save_counter, args):
    num_frames = (j+1)*args.num_processes                   # num frames
    return num_frames//args.save_interval > save_counter 


def make_env(env_id, seed, rank, num_agents, dist_threshold, arena_size, identity_size, num_steps, diff_reward, same_color, random_leader_name, goal_at_top, hide_goal, video_format):
    def _thunk():
        env = make_multiagent_env(env_id, num_agents, dist_threshold, arena_size, identity_size, num_steps, diff_reward, same_color, random_leader_name, goal_at_top, hide_goal, video_format)
        env.seed(seed + rank) # seed not implemented
        return env
    return _thunk


def make_multiagent_env(env_id, num_agents, dist_threshold, arena_size, identity_size, num_steps=128, diff_reward=True, same_color = False, random_leader_name=False, goal_at_top=False, hide_goal=False, video_format='webm'):
    scenario = scenarios.load(env_id+".py").Scenario(num_agents=num_agents, dist_threshold=dist_threshold,
                                                     arena_size=arena_size, identity_size=identity_size, num_steps=num_steps, diff_reward=diff_reward, same_color = same_color, random_leader_name=random_leader_name, goal_at_top=goal_at_top, hide_goal=hide_goal)
    world = scenario.make_world()
    
    env = MultiAgentEnv(world=world, 
                        reset_callback=scenario.reset_world, 
                        reward_callback=scenario.reward, 
                        observation_callback=scenario.observation,
                        info_callback=scenario.info if hasattr(scenario, 'info') else None,
                        done_callback=scenario.done,
                        discrete_action=True,
                        cam_range=arena_size,
                        num_steps=num_steps,
                        video_format=video_format,
                        )
    return env

    

def make_parallel_envs(args, goal_at_top = False):
    # make parallel environments
    if args.env_name == 'simpleFlockingAirsim':
        # envs = gym.make('simpleFlockingAirsim-v0')
        print('creating flocking env')
        envs = gym.make('simpleFlockingAirsim-v0')
        envs.world.max_steps_episode = args.num_steps_episode
        envs = PseudoVecEnvAirsim(envs)
    else:
        if args.num_processes > 1:
            envs = [make_env(args.env_name, args.seed, i, args.num_agents,args.dist_threshold, args.arena_size, args.identity_size, args.num_steps_episode, args.diff_reward, args.same_color, args.random_leader_name, goal_at_top, args.hide_goal, args.video_format) for i in range(args.num_processes)]
            envs = gym_vecenv.SubprocVecEnv(envs)
        else:
            envs = PsudoVecEnv(make_single_env(args, goal_at_top))

    if args.vec_normalize:
        envs = gym_vecenv.MultiAgentVecNormalize(envs, ob=False, ret=True)
    return envs


def make_single_env(args, goal_at_top = False):
    env = make_multiagent_env(args.env_name, args.num_agents, args.dist_threshold, args.arena_size, args.identity_size, args.num_steps_episode, args.diff_reward, args.same_color, args.random_leader_name, goal_at_top, args.hide_goal, args.video_format)
    return(env)

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

# for MAPE
class PsudoVecEnv():
    def __init__(self, env):
        self.env = env
        self.world = env.world
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.seed = env.seed
        self.n = env.n

        self.render = env.render
        self.startRecording = env.startRecording
        self.rgb2bgr = env.rgb2bgr
        self.recordFrame = env.recordFrame
        self.endVideo = env.endVideo
        self.saveFrame = env.saveFrame

    def reset(self):
        obs = self.env.reset()
        return np.array([obs])

    def step(self, a):
        obs, rew, done, info = self.env.step(a[0])
        if any(done):
            print('done', done)
            info['terminal_observation'] = obs
            obs = self.env.reset()
        return np.array([obs]), np.array([rew]), np.array([done]), [info]

# for airsim
class PseudoVecEnvAirsim():
    def __init__(self, env):
        self.env = env
        self.world = env.world
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.seed = env.seed
        self.n = env.n

    def reset(self):
        obs = self.env.reset()
        return np.array([obs])

    def step(self, a):
        obs, rew, done, info = self.env.step(a[0])
        if any(done):
            print('done', done)
            info['terminal_observation'] = obs
            obs = self.env.reset()
        return np.array([obs]), np.array([rew]), np.array([done]), [info]        