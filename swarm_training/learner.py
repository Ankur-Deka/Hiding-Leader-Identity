import numpy as np
import torch
import sys
# for path in sys.path:
#     print(path)
from .rlcore.algo import JointPPO
from .rlagent import Neo
from .mpnn import MPNN
from .utils import make_multiagent_env, make_parallel_envs
from copy import deepcopy

def setup_master(args, env=None, goal_at_top=False, return_env=False):
    if env is None:
        # env = make_multiagent_env(args.env_name, num_agents=args.num_agents, dist_threshold=args.dist_threshold, 
        #                           arena_size=args.arena_size, identity_size=args.identity_size, num_steps = args.num_steps_episode, diff_reward = args.diff_reward, same_color = args.same_color, random_leader_name = args.random_leader_name)
        args2 = deepcopy(args)
        args2.num_processes = 1
        env = make_parallel_envs(args2, goal_at_top)

    policy1 = None
    policy2 = None
    team1 = []
    team2 = []

    num_adversary = 0
    num_friendly = args.num_agents
    dim_p = env.world.dim_p
    # for i,agent in enumerate(env.world.policy_agents):
    #     if hasattr(agent, 'adversary') and agent.adversary:
    #         num_adversary += 1
    #     else:
    #         num_friendly += 1

    # share a common policy in a team
    action_space = env.action_space[0]
    entity_mp = args.entity_mp
    if args.env_name == 'simple_spread':
        num_entities = args.num_agents
    elif args.env_name == 'simple_formation':
        num_entities = 1
    elif args.env_name == 'simple_line':
        num_entities = 2
    elif args.env_name in ['simple_flocking', 'simple_trajectory', 'simple_waypoints', 'simple_dual_waypoints', 'simpleFlockingAirsim']:
        num_entities = 0
    else:
        raise NotImplementedError('Unknown environment, define entity_mp for this!')
    
    if entity_mp:
        pol_obs_dim = env.observation_space[0].shape[0] - 2*num_entities
    else:
        pol_obs_dim = env.observation_space[0].shape[0]

    # index at which agent's position is present in its observation
    pos_index = args.identity_size + 2
    
    if args.algo == 'scripted':
        master = scriptedMaster()
    elif args.algo == 'genetic' or args.algo == 'genetic_random':
        master = geneticMaster()
    elif args.algo == 'ppo':
        for i in range(args.num_agents):#, agent in enumerate(env.world.policy_agents):
            obs_dim = env.observation_space[0].shape[0]

            # if hasattr(agent, 'adversary') and agent.adversary:
            #     if policy1 is None:
            #         policy1 = MPNN(input_size=pol_obs_dim,num_agents=num_adversary,num_entities=num_entities,action_space=action_space,
            #                        pos_index=pos_index, mask_dist=args.mask_dist,entity_mp=entity_mp).to(args.device)
            #     team1.append(Neo(args,policy1,(obs_dim,),action_space))
            # else:
            if policy2 is None:
                policy2 = MPNN(input_size=pol_obs_dim,num_agents=num_friendly,num_entities=num_entities,dim_p=dim_p, action_space=action_space,
                       pos_index=pos_index, mask_dist=args.mask_dist,entity_mp=entity_mp).to(args.device)
            team2.append(Neo(args,policy2,(obs_dim,),action_space))
        master = Learner(args, [team1, team2], [policy1, policy2], env)
    else:
        print('PLEASE PROVIDE VALID ALGORITHM NAME') 
        # if args.continue_training:
        #     print("Loading pretrained model")
        #     master.load_models(torch.load(args.load_dir)['models'])

    if return_env:
        return master, env
    return master

class geneticMaster(object):
    def __init__(self, dims = 2):
        self.pos2v_scale = 0.5
        self.dims = dims
        

        # leader specific params
        self.v_leader = 1       
        self.leader_sep_weight = 0.5 
        self.leader_ali_weight = 0.5
        self.leader_coh_weight = 0.5
        self.leader_sep_max_cutoff = 2
        self.leader_ali_radius = 2
        self.leader_coh_radius = 2

        # follower specific params
        self.sep_weight = 0.5
        self.ali_weight = 0.5
        self.coh_weight = 0.5
        self.sep_max_cutoff = 2 
        self.ali_radius = 2
        self.coh_radius = 2

    def normalize(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    # get sep velocity for ith agent
    def get_sep_velocity(self, obs, i, is_leader=False):
        _, num_agents, obs_dim = obs.shape  # num_process=1, num_agents, obs_dim
        v_sep, c = 0, 0
        radius = self.leader_sep_max_cutoff if is_leader else self.sep_max_cutoff
        for j in range(num_agents):
            if i==j:
                continue
            diff = obs[0,i,self.dims:2*self.dims] - obs[0,j,self.dims:2*self.dims]
            dist = np.linalg.norm(diff)
            if 0 < dist < radius:
                v_sep += diff/dist
                c+=1
        if c:   # at least one neighbor affecting
            v_sep /= c
            v_sep = self.normalize(v_sep)*self.pos2v_scale
        return v_sep

    def get_ali_velocity(self, obs, i, is_leader=False):
        _, num_agents, obs_dim = obs.shape  # num_process=1, num_agents, obs_dim
        v_neighbor, c = 0, 0
        pos = obs[0,i,self.dims:2*self.dims]
        radius = self.leader_ali_radius if is_leader else self.ali_radius
        for j in range(num_agents):
            if i==j:
                continue
            neighbor_pos = obs[0,j,self.dims:2*self.dims]
            dist = np.linalg.norm(pos - neighbor_pos)
            if dist < radius:
                v_neighbor += obs[0,j,:self.dims]
                c += 1
        if c:
            v_neighbor /= c
            
        return v_neighbor
    def get_coh_velocity(self, obs, i, is_leader=False):
        _, num_agents, obs_dim = obs.shape  # num_process=1, num_agents, obs_dim
        p_neighbor, c = 0, 0
        pos = obs[0,i,self.dims:2*self.dims]
        radius = self.leader_coh_radius if is_leader else self.coh_radius
        for j in range(num_agents):
            if i==j:
                continue
            neighbor_pos = obs[0,j,self.dims:2*self.dims]
            dist = np.linalg.norm(pos - neighbor_pos)
            if dist < radius:
                p_neighbor += obs[0,j,self.dims:2*self.dims]
                c += 1
        if c == 0:
            return 0
        else:
            p_neighbor /= c
            pos_coh = p_neighbor - pos
            v_coh = self.normalize(pos_coh) * self.pos2v_scale
            return v_coh        

    # goal velocity only for the leader
    def get_goal_velocity(self, obs):
        pos_goal = obs[0,0,2*self.dims:3*self.dims] # already in relative coordinate
        v_goal = self.normalize(pos_goal)*self.v_leader
        return v_goal
        

    def discretize_action(self, action, th=0.02):
        assert action.shape[0] == self.dims, 'action doesn\'t match dimension'
        if max(abs(action))<th:
            discrete_action=0
        elif self.dims == 2:
            if abs(action[0])>abs(action[1]):
                discrete_action = 1 if action[0]<0 else 2
            else:
                discrete_action = 3 if action[1]<0 else 4
        elif self.dims == 3:
            direc = np.argmax(np.abs(action))    # 0,1 or 2
            discrete_action = direc*2+1 if action[direc]<0 else direc*2+2
        return discrete_action


    def __set_params(self, params):
        self.v_leader = params[0]
        self.leader_sep_w = params[1]
        self.leader_ali_w = params[2]
        self.leader_coh_w = params[3]
        self.leader_sep_max_cutoff = params[4]
        self.leader_ali_r = params[5]
        self.leader_coh_r = params[6]
        self.sep_w = params[7]
        self.ali_w = params[8]
        self.coh_w = params[9]
        self.sep_max_cutoff = params[10]
        self.ali_r = params[11]
        self.coh_r = params[12]


    def eval_act(self, obs, params):
        self.__set_params(params)

        _, num_agents, obs_dim = obs.shape  # num_process=1, num_agents, obs_dim
        actions_cts = []

        # get actions for leader
        v_goal = self.get_goal_velocity(obs)
        v_sep = self.get_sep_velocity(obs, 0, is_leader=True)
        v_ali = self.get_ali_velocity(obs, 0, is_leader=True)
        v_coh = self.get_coh_velocity(obs, 0, is_leader=True)
        v_desired = v_goal + self.leader_sep_weight*v_sep + self.leader_ali_weight*v_ali + self.leader_coh_weight*v_coh
        actions_cts.append(v_desired)

        # get actions for followers
        for i in range(1, num_agents):
            v_sep = self.get_sep_velocity(obs, i)
            v_ali = self.get_ali_velocity(obs, i)
            v_coh = self.get_coh_velocity(obs, i)

            v_desired = self.sep_weight*v_sep + self.ali_weight*v_ali + self.coh_weight*v_coh
            actions_cts.append(v_desired)

        actions = np.array([self.discretize_action(action) for action in actions_cts]).reshape(1,-1,1)
        
        return actions

            
class scriptedMaster(object):
    def __init__(self, dims = 2):
        self.dims = dims 

    def eval_act(self, obs, recurrent_hidden_states, mask):
        # used only while evaluating policies. Assuming that agents are in order of team & only a single process
        _, num_agents, obs_dim = obs.shape

        obs = obs[0]
        # leader actions
        goal_loc_rel = obs[0,-self.dims:]
        vel = obs[0,:self.dims]
        p = 1
        d = 0.1
        action = p*goal_loc_rel-d*vel
        leader_action = self.discretize_action(action)

        # print('\nLearner.py')
        # print('goal_loc_rel', goal_loc_rel)
        # print('vel', vel)
        # print('action', action)
        # print('leader_action', leader_action)


        # follower action
        p = 1
        d = 0.1
        leader_vel, leader_loc = obs[0,:self.dims], obs[0,self.dims:2*self.dims]
        followers_vels, followers_locs = obs[1:,:self.dims], obs[1:,self.dims:2*self.dims]
        e, e_dot = followers_locs-leader_loc, followers_vels-leader_vel
        followers_actions = -p*e-d*e_dot
        # print('follower cts actions')
        # print(followers_actions)
        followers_actions = [self.discretize_action(action) for action in followers_actions]
        # followers_actions = [0 for i in followers_actions]
        # followers_actions = np.zeros(num_agents-1)
        # combine
        actions = np.array([leader_action]+followers_actions).reshape(1,-1,1)
        # print('actions')
        # print(actions)
        return actions

        # obs = obs[0]
        # obs1 = []
        # obs2 = []
        # all_obs = []
        # for i in range(len(obs)):
        #     agent = self.env.world.policy_agents[i]
        #     if hasattr(agent, 'adversary') and agent.adversary:
        #         obs1.append(torch.as_tensor(obs[i],dtype=torch.float,device=self.device).view(1,-1))
        #     else:
        #         obs2.append(torch.as_tensor(obs[i],dtype=torch.float,device=self.device).view(1,-1))
        # if len(obs1)!=0:
        #     all_obs.append(obs1)
        # if len(obs2)!=0:
        #     all_obs.append(obs2)

        # actions = []
        # for team,policy,obs in zip(self.teams_list,self.policies_list,all_obs):
        #     if len(obs)!=0:
        #         _,action,_,_ = policy.act(torch.cat(obs).to(self.device),None,None,deterministic=True)
        #         actions.append(action.squeeze(1).cpu().numpy())

        # return np.hstack(actions).reshape(1,-1)

    def discretize_action(self, action, th=0.02):
        assert action.shape[0] == self.dims, 'action doesn\'t match dimension'
        if max(abs(action))<th:
            discrete_action=0
        elif self.dims == 2:
            if abs(action[0])>abs(action[1]):
                discrete_action = 1 if action[0]<0 else 2
            else:
                discrete_action = 3 if action[1]<0 else 4
        elif self.dims == 3:
            direc = np.argmax(np.abs(action))    # 0,1 or 2
            discrete_action = direc*2+1 if action[direc]<0 else direc*2+2

        return discrete_action

class Learner(object):
    # supports centralized training of agents in a team
    def __init__(self, args, teams_list, policies_list, env):
        self.teams_list = [x for x in teams_list if len(x)!=0]
        self.all_agents = [agent for team in teams_list for agent in team]
        self.policies_list = [x for x in policies_list if x is not None]
        self.trainers_list = [JointPPO(policy, args.clip_param, args.num_updates, args.num_mini_batch, args.value_loss_coef,
                                       args.entropy_coef, lr=args.lr, max_grad_norm=args.max_grad_norm,
                                       use_clipped_value_loss=args.clipped_value_loss) for policy in self.policies_list]
        self.num_processes = args.num_processes
        self.device = args.device
        self.env = env 
        self.step = 0         # step in rollout storage, env step may be completely different
        self.num_steps = args.num_steps_buffer

        for i, agent in enumerate(self.all_agents):
            agent.rollouts.to(self.device)


    @property
    def all_policies(self):
        return [agent.actor_critic.state_dict() for agent in self.all_agents]

    @property
    def team_attn(self):
        return self.policies_list[0].attn_mat

    def initialize_obs(self, obs):
        # obs - num_processes x num_agents x obs_dim
        for i, agent in enumerate(self.all_agents):
            agent.initialize_obs(torch.from_numpy(obs[:,i,:]).float().to(self.device))
            agent.rollouts.to(self.device)

    def act(self, step = None):
        if step is None:
            step = self.step
        actions_list = []
        for team, policy in zip(self.teams_list, self.policies_list):
            # concatenate all inputs
            all_obs = torch.cat([agent.rollouts.obs[step] for agent in team])
            all_hidden = torch.cat([agent.rollouts.recurrent_hidden_states[step] for agent in team])
            all_masks = torch.cat([agent.rollouts.masks[step] for agent in team])

            props = policy.act(all_obs, all_hidden, all_masks, deterministic=False) # a single forward pass 
            # split all outputs
            n = len(team)
            all_value, all_action, all_action_log_prob, all_states = [torch.chunk(x, n) for x in props]
            for i in range(n):
                team[i].value = all_value[i]
                team[i].action = all_action[i]
                team[i].action_log_prob = all_action_log_prob[i]
                team[i].states = all_states[i]
                actions_list.append(all_action[i].cpu().numpy())

        return actions_list

    def update(self, return_rew = False):
        # treat as pseudo terminal step, compute terminal values for all agents, will automatically have no effect if already at terminal state of episode
        step = (self.step-1)%self.num_steps
        with torch.no_grad():
            for team, policy in zip(self.teams_list, self.policies_list):
                # concatenate all inputs
                all_obs_nxt = torch.cat([agent.rollouts.obs_nxt[step] for agent in team])
                # all_hidden = torch.cat([agent.rollouts.recurrent_hidden_states[step] for agent in team])
                # all_masks = torch.cat([agent.rollouts.masks[step] for agent in team])

                n = len(team)
                vals_nxt_team = policy.get_value(all_obs_nxt) # a single forward pass
                vals_nxt_team = torch.chunk(vals_nxt_team, n)

                for i, agent in enumerate(team):
                    val_nxt = vals_nxt_team[i]
                    agent.terminate_episodes(step, val_nxt)


        return_vals = []
        # use joint ppo for training each team
        for i, trainer in enumerate(self.trainers_list):
            rollouts_list = [agent.rollouts for agent in self.teams_list[i]]
            vals = trainer.update(rollouts_list)
            # -------- get avg rewards in buffer -------- #
            vals = np.array(vals).reshape((1,3)).repeat(len(rollouts_list), axis=0)
            
            if return_rew: 
                rewards = np.array([agent.rollouts.rewards.to('cpu').numpy().mean() for agent in self.teams_list[i]]).reshape(-1,1)
                vals = np.concatenate((vals, rewards), axis = 1)
            
            return_vals.append(vals)
        
        return np.vstack(return_vals)

    # remove these!
    # def wrap_horizon(self):
    #     for team, policy in zip(self.teams_list,self.policies_list):
    #         last_obs = torch.cat([agent.rollouts.obs[-1] for agent in team])
    #         last_hidden = torch.cat([agent.rollouts.recurrent_hidden_states[-1] for agent in team])
    #         last_masks = torch.cat([agent.rollouts.masks[-1] for agent in team])
            
    #         with torch.no_grad():
    #             next_value = policy.get_value(last_obs, last_hidden, last_masks)

    #         all_value = torch.chunk(next_value,len(team))
    #         for i in range(len(team)):
    #             team[i].wrap_horizon(all_value[i])

    # def after_update(self):
    #     for agent in self.all_agents:
    #         agent.after_update()

    # insert before step
    def update_obs(self, obs):
        obs = torch.from_numpy(obs).float().to(self.device) # [num_processes, num_agents, obs_dim]
        for i, agent in enumerate(self.all_agents):
            agent_obs = obs[:, i, :]        # all processes for ith agent
            agent.update_obs(agent_obs)
        # print('data in buffer update_obs')
        # for i, agent in enumerate(self.all_agents):
        #     obs = agent.rollouts.obs[self.step].to('cpu').numpy()
        #     actions= agent.rollouts.actions[self.step].to('cpu').numpy()
        #     rewards = agent.rollouts.rewards[self.step].to('cpu').numpy()
        #     obs_nxt = agent.rollouts.obs_nxt[self.step].to('cpu').numpy()
        #     print('agent', i, obs, actions, rewards, obs_nxt)

    # insert after step reward, masks, obs_nxt, info
    def update_obs_nxt(self, reward, masks, obs_nxt, info, auto_terminate=True):
        obs_nxt = torch.from_numpy(obs_nxt).float().to(self.device)
        done = torch.FloatTensor([[info_env['env_done']] for info_env in info]).to(self.device) # consider done only when env_done. Varies across parallel envs, fixed for all agents in one env
        for i, agent in enumerate(self.all_agents):
            agent_obs_nxt = obs_nxt[:, i, :]
            agent.update_obs_nxt(reward[:,i].unsqueeze(1), masks[:,i].unsqueeze(1), agent_obs_nxt, done, auto_terminate)

        # print('data in buffer')
        # for i, agent in enumerate(self.all_agents):
        #     obs = agent.rollouts.obs[self.step].to('cpu').numpy()
        #     actions= agent.rollouts.actions[self.step].to('cpu').numpy()
        #     rewards = agent.rollouts.rewards[self.step].to('cpu').numpy()
        #     obs_nxt = agent.rollouts.obs_nxt[self.step].to('cpu').numpy()
        #     print('agent', i, obs, actions, rewards, obs_nxt)
        self.step = (self.step + 1) % self.num_steps

    # for actual termination, not early terminate 
    def terminate_episodes(self, val_nxt=None):
        step = (self.step-1)%self.num_steps
        if val_nxt is None:
            val_nxt = torch.zeros((self.num_processes, self.env.n), dtype=torch.float).to(self.device)   # [num_process, num_agents]
        for i, agent in enumerate(self.all_agents):
            agent.terminate_episodes(step, val_nxt[:,i])
        # need to look at counters


    def add_to_reward(self, rewards):
        # rewards: [num_processes, num_steps, num_agents], num_steps should be last episode's length
        rewards = torch.from_numpy(rewards).float().to(self.device).permute(2, 0, 1)  # [num_agents, num_processes, num_steps]
        
        for agent, agent_rewards in zip(self.all_agents, rewards):
            agent.add_to_reward(agent_rewards)

    def load_models(self, policies_list):
        for agent, policy in zip(self.all_agents, policies_list):
            agent.load_model(policy)

    def eval_act(self, obs, recurrent_hidden_states, mask):
        # used only while evaluating policies. Assuming that agents are in order of team!
        obs = obs[0]
        obs1 = []
        obs2 = []
        all_obs = []
        for i in range(len(obs)):
            # agent = self.env.world.policy_agents[i]
            # if hasattr(agent, 'adversary') and agent.adversary:
            #     obs1.append(torch.as_tensor(obs[i],dtype=torch.float,device=self.device).view(1,-1))
            # else:
            obs2.append(torch.as_tensor(obs[i],dtype=torch.float,device=self.device).view(1,-1))
        if len(obs1)!=0:
            all_obs.append(obs1)
        if len(obs2)!=0:
            all_obs.append(obs2)

        actions = []
        for team,policy,obs in zip(self.teams_list,self.policies_list,all_obs):
            if len(obs)!=0:
                _,action,_,_ = policy.act(torch.cat(obs).to(self.device),None,None,deterministic=True)
                actions.append(action.squeeze(1).cpu().numpy())

        return np.hstack(actions).reshape(1,-1)

    def set_eval_mode(self):
        for agent in self.all_agents:
            agent.actor_critic.eval()

    def set_train_mode(self):
        for agent in self.all_agents:
            agent.actor_critic.train()
