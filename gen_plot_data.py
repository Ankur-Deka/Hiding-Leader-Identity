import os, sys, shutil
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from copy import deepcopy
import pandas as pd

from joint_main import jointController
from joint_arguments import get_args


np.set_printoptions(suppress=True, precision=4)


def extract_ckpts(args):
    ckpts_path = os.path.join(args.root_save_dir, 'run_'+str(args.load_run))
    all_ckpts = np.sort([int(ckpt[5:-3]) for ckpt in os.listdir(ckpts_path) if ckpt.endswith('.pt')])
    # min_ckpt, max_ckpt = all_ckpts[0], all_ckpts[-1]
    # ckpts = np.linspace(min_ckpt, max_ckpt, args.num_cross_eval_ckpts).astype(int)
    return all_ckpts
    
def viz_rewards(task_rewards, privacy_rewards):
    plt.subplot(1,2,1)
    plt.imshow(task_rewards)
    plt.grid(False)
    plt.colorbar()
    plt.title('Task Rewards')
    plt.xlabel('Adversary Checkpoint')
    plt.ylabel('Swarm Checkpoint')
    
    plt.subplot(1,2,2)
    plt.imshow(privacy_rewards)
    plt.grid(False)
    plt.colorbar()
    plt.title('Privacy Rewards')
    plt.xlabel('Adversary Checkpoint')
    plt.ylabel('Swarm Checkpoint')

    plt.savefig(os.path.join(args.out_dir, 'cross_evaluation.png'), dpi=300)

if __name__ == '__main__':
    args = get_args()
    if args.seed is None:
        args.seed = np.random.randint(0,10000)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    myController = jointController(args)
    
    pprint(vars(args))

    ckpts = extract_ckpts(args)
    print(ckpts)
    metrics_dict = {'Method': [], 'No. of environment steps': [], 'Task reward': [], 'Privacy reward': []}
    # -------- loading pretrained models -------- #
    swarm_checkpoint, adversary_checkpoint = None, None
    ckpts_path = os.path.join(args.root_save_dir, 'run_'+str(args.load_run))
    
    
    # task_rewards = np.zeros((args.num_cross_eval_ckpts, args.num_cross_eval_ckpts))
    # privacy_rewards = np.zeros((args.num_cross_eval_ckpts, args.num_cross_eval_ckpts))
    
    for i,ckpt in enumerate(ckpts):
        swarm_load_path = os.path.join(ckpts_path, 'ckpt_{}.pt'.format(ckpt))
        swarm_checkpoint = torch.load(swarm_load_path)['swarm']
        adversary_load_path = os.path.join(ckpts_path, 'ckpt_{}.pt'.format(ckpt))
        adversary_checkpoint = torch.load(adversary_load_path)['adversary']
        metrics = myController.evaluate(np.random.randint(0,10000), swarm_checkpoint, adversary_checkpoint)

        task_rewards = metrics['episode_task_rewards'] # num_episodesxnum_agents
        privacy_rewards = metrics['episode_privacy_rewards'] #num_ep, num_steps, num_agents
        print(ckpt, task_rewards)
        for team_task_rew, team_priv_reward in zip(task_rewards, privacy_rewards):
            metrics_dict['Method'].append('Ours')
            metrics_dict['No. of environment steps'].append(ckpt*args.save_interval)
            metrics_dict['Task reward'].append(team_task_rew.mean()/args.num_steps_episode)
            metrics_dict['Privacy reward'].append(1-team_priv_reward[-1].mean()/args.num_steps_episode) # use only the last tiem step!

    df = pd.DataFrame(metrics_dict)
    df.to_csv('Plotting_data_234.csv')
    # print(task_rewards)
    # print(privacy_rewards)
    # viz_rewards(task_rewards, privacy_rewards)
    

    # print('Metrics \n{}'.format(metrics))

    #  , 'ckpt_{}.pt'.format(args.load_ckpt))



    # if args.load_mode == 'joint':
    #     print('args.load_path', args.load_path)
    #     
    #     adversary_checkpoint = checkpoint['adversary'] if args.use_adversary else None
    # elif args.load_mode == 'individual':
    #     if not args.swarm_load_path is None:
    #         swarm_checkpoint = torch.load(args.swarm_load_path)
    #         if 'swarm' in swarm_checkpoint.keys():
    #             swarm_checkpoint = swarm_checkpoint['swarm']
    #     if args.use_adversary:
    #         adversary_checkpoint = torch.load(args.adversary_load_path)
    #         print('args.adversary_load_path', args.adversary_load_path)
    #         if 'adversary' in adversary_checkpoint.keys():
    #             adversary_checkpoint = adversary_checkpoint['adversary']
    #     else:
    #         adversary_checkpoint = None
    #     print('loaded swarm and adversary', 'joint_main.py')

    
    # # Save configs
    # if not args.out_dir is None:
    #     with open(os.path.join(args.out_dir, 'params.json'), 'w') as f:
    #         params = deepcopy(vars(args))
    #         params.pop('device')
    #         json.dump(params, f)
    #     # save to common file
    #     with open(os.path.join('output','all_config.txt'),'a') as f:
    #         f.write('\n\n'+str(params))
    


    # metrics = myController.evaluate(args.seed, swarm_checkpoint, adversary_checkpoint)
    # print('Metrics \n{}'.format(metrics))
    # 