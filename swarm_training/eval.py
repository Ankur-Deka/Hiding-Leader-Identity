import sys, os
sys.path.append('../mape')
import numpy as np
import torch
from arguments import get_args
from utils import normalize_obs, extract_data
from learner import setup_master
import time
import json
from copy import deepcopy

def evaluate(args, seed, policies_list, ob_rms=None, render=False, env=None, master=None, render_attn=True):
    """
    RL evaluation: supports eval through training code as well as independently
    policies_list should be a list of policies of all the agents;
    len(policies_list) = num agents
    """
    if env is None or master is None: # if any one of them is None, generate both of them
        master, env = setup_master(args, return_env=True)

    if seed is None: # ensure env eval seed is different from training seed
        seed = np.random.randint(0,100000)
    print("Evaluation Seed: ",seed)


    env.seed(seed)

    if ob_rms is not None:
        obs_mean, obs_std = ob_rms
    else:
        obs_mean = None
        obs_std = None

    master.load_models(policies_list)
    master.set_eval_mode()

    num_eval_episodes = args.num_eval_episodes
    all_episode_rewards = np.full((num_eval_episodes, env.n), 0.0)
    per_step_rewards = np.full((num_eval_episodes, env.n), 0.0)

    # TODO: provide support for recurrent policies and mask
    recurrent_hidden_states = None
    mask = None

    # world.dists at the end of episode for simple_spread
    final_min_dists = []
    leader_names = []
    num_success = 0
    episode_length = 0

    # exporting trajectories
    # if not args.out_dir is None:
    #     path = os.path.join(args.out_dir, 'traj')
    #     if not os.path.exists(path):
    #         os.makedirs(path)

    for t in range(num_eval_episodes):
        obs = env.reset() # although it also auto resets
        leader_names.append({'episode':t, 'leaderName':env.world.leader_name})

# data = [{episode:0, leaderID: 0},
#         {episode:5, leaderID: 4}]
        # -------- recording video -------- #
        if args.record_video:
            video_name = str(args.load_run)+'_'+str(args.load_ckpt)+'_'+str(t)+'.webm' if args.store_video_together else str(t)+'.webm' 
            video_path = os.path.join(args.video_path, video_name)
            env.startRecording(video_path)

        # ----- for exporting positions to file ------ #
        traj = [extract_data(obs)]
        
        obs = normalize_obs(obs, obs_mean, obs_std)
        done = [False]*env.n
        episode_rewards = np.zeros((1,env.n))
        episode_steps = 0
        if render:
            attn = None if not render_attn else master.team_attn
            if attn is not None and len(attn.shape)==3:
                attn = attn.max(0)
            env.render(attn=attn)
        i = 0
        while not np.all(done):
            print(i)
            i+=1
            actions = []
            with torch.no_grad():
                actions = master.eval_act(obs, recurrent_hidden_states, mask)
            episode_steps += 1
            obs, reward, done, info = env.step(actions)
            traj.append(extract_data(obs))
            obs = normalize_obs(obs, obs_mean, obs_std)
            episode_rewards += np.array(reward)
            if render:
                time.sleep(0.1)
                attn = None if not render_attn else master.team_attn
                if attn is not None and len(attn.shape)==3:
                    attn = attn.max(0)
                env.render(attn=attn)

            if args.record_video:
                env.recordFrame()
            #     time.sleep(0.08)

        per_step_rewards[t] = episode_rewards/episode_steps

        num_success += info[0]['is_success']
        episode_length = (episode_length*t + info['n'][0]['world_steps'])/(t+1)

        # for simple spread env only
        if args.env_name == 'simple_spread':
            final_min_dists.append(env.world.min_dists)
        elif args.env_name == 'simple_formation' or args.env_name=='simple_line':
            final_min_dists.append(env.world.dists)

        if render:
            print("Ep {} | Success: {} \n Av per-step reward: {:.2f} | Ep Length {}".format(t,info['n'][0]['is_success'],
                per_step_rewards[t][0],info['n'][0]['world_steps']))
        all_episode_rewards[t, :] = episode_rewards # all_episode_rewards shape: num_eval_episodes x num agents

        if args.record_video:
            env.endVideo()

        if not args.out_dir is None:
            traj = np.vstack(traj).astype(np.float16)
            np.save(os.path.join(args.out_dir, 'traj', str(t)),traj)
    
    if args.record_video:
        with open(os.path.join(args.out_dir, 'leader_names_in_video.txt'), 'w') as f:
            f.write(str(leader_names))

    return all_episode_rewards, per_step_rewards, final_min_dists, num_success, episode_length


if __name__ == '__main__':
    args = get_args()
    checkpoint = torch.load(args.load_dir, map_location=lambda storage, loc: storage)
    policies_list = checkpoint['models']
    
    if args.vec_normalize:
        ob_rms = checkpoint['ob_rms']
    else:
        ob_rms = None

    if args.seed is None: # ensure env eval seed is different from training seed
        args.seed = np.random.randint(0,100000)
        
    # Save configs
    if not args.out_dir is None:
        with open(os.path.join(args.out_dir, 'params.json'), 'w') as f:
            params = deepcopy(vars(args))
            params.pop('device')
            json.dump(params, f)

        # save to common file
        with open(os.path.join('output','all_config.txt'),'a') as f:
            f.write('\n\n'+str(params))


    all_episode_rewards, per_step_rewards, final_min_dists, num_success, episode_length = evaluate(args, args.seed, 
                    policies_list, ob_rms, args.render, render_attn=args.masking)
    print("Average Per Step Reward {}\nNum Success {}/{} | Av. Episode Length {:.2f})"
            .format(per_step_rewards.mean(0),num_success,args.num_eval_episodes,episode_length))
    if final_min_dists:
        print("Final Min Dists {}".format(np.stack(final_min_dists).mean(0)))
