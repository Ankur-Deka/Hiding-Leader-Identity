import sys
sys.path.append('../mape')
import os
import json
import datetime
import numpy as np
import torch
import utils
from utils import should_update, should_eval, should_save
import random
from copy import deepcopy
from arguments import get_args
from tensorboardX import SummaryWriter
from eval import evaluate
from learner import setup_master
from pprint import pprint
import time

np.set_printoptions(suppress=True, precision=4)


def train(args, return_early=False):
    # -------- setup master -------- #
    writer = SummaryWriter(args.log_dir)    
    envs = utils.make_parallel_envs(args, auto_reset = True) 
    master = setup_master(args) 

    # -------- used during evaluation only -------- #
    eval_master, eval_env = setup_master(args, auto_reset = True, return_env=True)
    
    # -------- initialize logistics -------- #
    obs = envs.reset() # shape - num_processes x num_agents x obs_dim
     
    n = len(master.all_agents)
    episode_rewards = torch.zeros([args.num_processes, n], device=args.device)
    final_rewards = torch.zeros([args.num_processes, n], device=args.device)
    update_counter, eval_counter, save_counter = 0, 0, 0     # no. of master updates/evaluations/saves

    # -------- start simulations -------- #
    start = datetime.datetime.now()

    for j in range(args.num_iters):         # each j update -> 1 step in all parallel envs
        # -------------------- update observation -------------------- #
        if args.render and args.num_processes==1:
            envs.render()
            # time.sleep(0.1)
        master.update_obs(obs)

        # -------- get actions -------- #
        with torch.no_grad():
            actions_list = master.act()  ## act takes in step (rollout storage step) # actions_list is [num_agents, num_processes, act_dim]

        agent_actions = np.transpose(np.array(actions_list),(1,0,2))  # [num_processes, num_agents, act_dim]
        
        # -------- step through vec env -------- #
        # obs: [num_processes, num_agents, obs_dim]
        # reward, done: [num_processes, num_agents]
        # info: [dict, dict, .. ]. list of dicts for parallel envs, one dict per env
        # obs in global coordinate (except landmarks are viewed in respective agent frames)
        obs_nxt, reward, done, info = envs.step(agent_actions) 
        # for k,i in enumerate(info):
        #     if i['env_done']:
        #         print('env', k, 'done at', j)
        #     if i['is_success']:
        #         print('env', k, 'success at', j)

        reward = torch.from_numpy(np.stack(reward)).float().to(args.device) 
        # print('data in main', 'obs', obs, 'actions', agent_actions, 'reward', reward, 'obs_nxt', obs_nxt)
        episode_rewards += reward
        masks = torch.FloatTensor(1-1.0*done).to(args.device)

        # -------------------- update rollout, now includes wrap env -------------------- #
        master.update_obs_nxt(reward, masks, obs_nxt, info)     # computes returns internally
        
        obs = deepcopy(obs_nxt)

        # -------------------- algorithm update -------------------- #
        if should_update(j, update_counter, args):
            # run update for num_updates times
            return_rew = args.algo=='ppo'
            return_vals = master.update(return_rew = return_rew)   # internally considers pseudo early termination of all episodes
            value_loss = return_vals[:, 0]
            action_loss = return_vals[:, 1]
            dist_entropy = return_vals[:, 2]

            if return_rew:
                buffer_avg_reward = return_vals[:, 3]  
                print('Buffer avg per step reward {}'.format(buffer_avg_reward))
                # tensorboard 
                for idx,rew in enumerate(buffer_avg_reward):
                    num_env_steps = (j+1)*args.num_processes
                    writer.add_scalar('train_buffer_avg_per_step_reward/'+'agent'+str(idx), rew, num_env_steps)

            update_counter += 1


        # -------- saving checkpoints -------- #
        if should_save(j, save_counter, args) and not args.test:
            savedict = {'models': [agent.actor_critic.state_dict() for agent in master.all_agents]}
            ob_rms = (envs.ob_rms[0].mean, envs.ob_rms[0].var) if args.vec_normalize and envs.ob_rms is not None else (None,None) 
            savedict['ob_rms'] = ob_rms
            savedir = args.save_dir+'/ckpt_'+str(save_counter)+'.pt'
            torch.save(savedict, savedir)
            
            # store mappping from ckpt to env steps
            num_env_steps = (j+1)*args.num_processes
            f = open(os.path.join(args.save_dir, 'ckpt_to_numEnvSteps.txt'),'a')
            f.write('\n'+str(save_counter)+' '+str(num_env_steps))
            f.close()

            save_counter+=1


        # -------- evaluating performance -------- #
        # if should_eval(j, eval_counter, args):
        #     ob_rms = (envs.ob_rms[0].mean, envs.ob_rms[0].var) if args.vec_normalize and envs.ob_rms is not None else (None,None)
        #     print('===========================================================================================')
        #     num_env_steps = (j+1)*args.num_processes
        #     print('Evaluation no: {}, Num env steps {}'.format(eval_counter, num_env_steps))
        #     eval_episode_rews, eval_perstep_rewards, final_min_dists, num_success, eval_episode_len = evaluate(args, None, master.all_policies,ob_rms=ob_rms, env=eval_env,master=eval_master)
        #     mean_ep_rew = eval_episode_rews.mean(axis=0)
        #     num_env_steps = (j+1)*args.num_processes
        #     print('Mean episode rewards {}'.format(mean_ep_rew))
        #     print('Mean per-step reward {:.2f}'.format(eval_perstep_rewards.mean()))
        #     print('Num success {:d}/{:d} | Episode Length {:.2f}'.format(num_success, args.num_eval_episodes, eval_episode_len))
        #     if final_min_dists:
        #         print('Final_dists_mean {}'.format(np.stack(final_min_dists).mean(0)))
        #         print('Final_dists_var {}'.format(np.stack(final_min_dists).var(0)))

        #     print('===========================================================================================\n')

        #     if not args.test:
        #         writer.add_scalar('all/eval_success', 100.0*num_success/args.num_eval_episodes, num_env_steps)
        #         writer.add_scalar('all/episode_length', eval_episode_len, num_env_steps)
        #         for idx in range(n):
        #             # writer.add_scalar('agent'+str(idx)+'/eval_per_step_reward', eval_perstep_rewards.mean(0)[idx], num_env_steps)
        #             writer.add_scalar('eval_episode_reward/'+'agent'+str(idx), mean_ep_rew[idx], num_env_steps)
        #             if final_min_dists:
        #                 writer.add_scalar('eval_min_dist''agent'+str(idx), np.stack(final_min_dists).mean(0)[idx], num_env_steps)

        #     curriculum_success_thres = 0.9
        #     if return_early and num_success*1./args.num_eval_episodes > curriculum_success_thres:
        #         savedict = {'models': [agent.actor_critic.state_dict() for agent in master.all_agents]}
        #         ob_rms = (None, None) if envs.ob_rms is None else (envs.ob_rms[0].mean, envs.ob_rms[0].var)
        #         savedict['ob_rms'] = ob_rms
        #         savedir = args.save_dir+'/ep'+str(j)+'.pt'
        #         torch.save(savedict, savedir)
        #         print('===========================================================================================\n')
        #         print('{} agents: training complete. Breaking.\n'.format(args.num_agents))
        #         print('===========================================================================================\n')
        #         break
        #     eval_counter += 1

    writer.close()
    if return_early:
        return savedir

if __name__ == '__main__':
    args = get_args()
    if args.seed is None:
        args.seed = random.randint(0,10000)
    args.num_iters = args.num_frames // args.num_processes
    torch.manual_seed(args.seed)
    # torch.set_num_threads(1)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    pprint(vars(args))
    if not args.test:
        with open(os.path.join(args.save_dir, 'params.json'), 'w') as f:
            params = deepcopy(vars(args))
            params.pop('device')
            json.dump(params, f)

        # save to common file
        with open(os.path.join(args.root_save_dir,'all_config.txt'),'a') as f:
            f.write('\n\n'+str(params))

    train(args)
