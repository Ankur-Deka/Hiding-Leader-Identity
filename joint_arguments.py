import argparse
import os
import sys
import torch
import shutil


def process_runID(runID, rootDir):
    if runID == 'latest':
        runID = max([int(d.split('_')[-1]) for d in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir,d))])
        runID = int(float(runID))
    elif runID == 'None' or runID == None:
        runID = None
    return runID

def process_ckpt(ckpt, rootDir, runID):
    if runID == None:
        ckpt = None
    elif ckpt == 'latest':
        run_dir = os.path.join(rootDir, 'run_'+str(runID))    
        print(run_dir)
        ckpt = max([int(file[5:][:-3]) for file in os.listdir(run_dir) if file.endswith('.pt')])
        ckpt = int(float(ckpt))
    return ckpt

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    
    # environment
    parser.add_argument('--env-name', default='simple_flocking', help='one from {simple_spread, simple_formation, simple_line, simple_flocking}, first 3 may not work)')
    parser.add_argument('--num-agents', type=int, default=6)
    parser.add_argument('--dist-threshold', type=float, default=0.12, help='dist threshold for goal reaching')
    parser.add_argument('--masking', action='store_true', help='restrict communication to within some threshold')
    parser.add_argument('--mask-dist', type=float, default=1.0, help='distance to restrict comms')
    parser.add_argument('--dropout-masking', action='store_true', help='dropout masking enabled')
    parser.add_argument('--entity-mp', action='store_true', help='enable entity message passing')
    parser.add_argument('--identity-size', default=0, type=int, help='size of identity vector')
    parser.add_argument('--num-steps-episode', type=int, default=50)
    parser.add_argument('--vec-normalize', action = 'store_true')    

    # training 
    parser.add_argument('--seed', type=int, default=None, help='random seed (default: None)')
    parser.add_argument('--num-processes', type=int, default=10, help='how many training CPU processes to use (default: 32)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--cuda-id', type=int, default=0, help='which cuda device?')
    parser.add_argument('--num-frames', type=int, default=int(1e6), help='number of frames to train (default: 50e6)')
    parser.add_argument('--arena-size', type=int, default=1, help='size of arena')

    # evaluation
    parser.add_argument('--num-eval-episodes', type=int, default=3, help='number of episodes to evaluate with')
    parser.add_argument('--num-cross-eval-ckpts', type=int, default=10, help='Only for cross evaluation across different checkpoints. Number of checkpoints to use.')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--record-video', action='store_true', default=False, help='record evaluation video')

    parser.add_argument('--video-format', type=str, default='gif', help='{webm, mp4, gif}')
    parser.add_argument('--store-traj', action='store_true')
    parser.add_argument('--plot-trajectories', action='store_true', help='do you want to plot trajectories during evaluation')


    # PPO/SAC
    parser.add_argument('--algo', default='ppo', help='algorithm to use: {ppo, scripted, genetic}')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.95, help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01, help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5, help='value loss coefficient (default: 0.05)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='max norm of gradients (default: 0.5)')
    parser.add_argument('--num-updates', type=int, default=4, help='number of updates (default: 4 for PPO, set automatically for SAC)')
    parser.add_argument('--num-mini-batch', type=int, default=32, help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip-param', type=float, default=0.2, help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--buffer-size', type=int, default=500, help='experience/replay buffer')
    parser.add_argument('--update-every', type=int, default=500, help='update after this much interval, set automatically for PPO')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size sampled from buffer for SAC iterations. automatically set for PPO')

    # Genetic 
    parser.add_argument('--population_size', type=int, default=10, help='population size for genetic algorithm')
    
    # logging
    # parser.add_argument('--test', action='store_true')
    parser.add_argument('--mode', type=str, default='test', help='One of {train, test}')
    parser.add_argument('--root-save-dir', default='./marlsave', help='Root directory to save models. Relative path is auto generated')
    parser.add_argument('--log-dir', default='logs', help='directory to save logs')
    parser.add_argument('--save-interval', type=int, default=500, help='save interval in terms of env steps')
    parser.add_argument('--log-interval', type=int, default=10, help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--load-path', default=None, type=str, help='path to load all policies from. either provide load-path or {load-run and load-ckpt}. If you provide both, load-dir will be used')
    parser.add_argument('--load-run', default=None, help='run to load all policies from, {a number or "latest"')
    parser.add_argument('--load-ckpt', default=None, help='checkpoint to load from, {a number or "latest"}')
    parser.add_argument('--eval-interval', default=2000, type=int)
    parser.add_argument('--continue-training', action='store_true')
    parser.add_argument('--diff-reward', type=int, default=1, help='{0,1} Whether to use difference in distance reward for swarm?')
    parser.add_argument('--goal-at-top', action='store_true')
    parser.add_argument('--same-color', action='store_true')
    parser.add_argument('--hide-goal', action='store_true')
    parser.add_argument('--random-leader-name', default=1, type=int, help='During evaluation, assigns a random name to leader. This is just a name, in agents list first agent is always the leader')
    parser.add_argument('--out-dir', default=None, type=str, help='Which folder do you wish to store the trajectories and videos in?. Final path is "output/out-dir"')
    parser.add_argument('--store-video-together', action='store_true', help='if passed, all videos are saved in "output/videos". Else videos are saved in "output/out-dir/videos"')
    parser.add_argument('--algo-stage', default=None, type=str, help="which algo_stage name should be saved in the file learder_names_in_videos.txt. It doesn't actually change the algo.")

    # adversary arguments
    parser.add_argument('--adversary-version', type=str, default='V1')
    parser.add_argument('--adversary-num-trajs', type=int, default=None, help='How many past trajectories should the adversary be trained on. Default is as many in buffer_size')
    parser.add_argument('--obs-dim', type=int, default=2, help='observation dimension for one agent, later automatically select it from env observation_space')
    parser.add_argument('--adversary-hidden-dim', type=int, default=12, help='Hidden dimension of LSTM or NN')
    parser.add_argument('--adversary-hidden-ch', type=int, default=16, help='Hidden channels for adversary V2')
    parser.add_argument('--adversary-apply-maxpool', type=bool, default=True, help='Whether to use maxpool for adversary V2')
    parser.add_argument('--adversary-init-steps', type=int, default=1, help='Initialization steps for adversary')
    parser.add_argument('--data-temp-dir', type=str, default='tmp/trajs', help='Temporary dir that stores trajectory files for training adversary')
    parser.add_argument('--privacy-reward', type=float, default=1, help='per step negative reward if leader is revealed')
    parser.add_argument('--adversary-num-epochs', type=int, default=4, help='Number of epochs for adversary')
    parser.add_argument('--adversary-lr', default=1e-3, help='learning rate for adversary')
    parser.add_argument('--adversary-genetic-beta', type=float, default=0.5, help='beta parameter for genetic adversary, check fitness function in joint_main_genetic.py')


    # what to train
    parser.add_argument('--train-swarm', type=int, default=1, help='train swarm?')
    parser.add_argument('--use-adversary', type=int, default=1, help='use adversary?')
    parser.add_argument('--train-adversary', type=int, default=1, help='train adversary?')
    # pretrained swarm and adversary paths
    parser.add_argument('--load-mode', default=None, help='{individual, joint, None}. Load swarm and adversary individually or jointly. don\'t load if None')
    parser.add_argument('--swarm-load-path', default=None, help='path to load all policies from')
    parser.add_argument('--swarm-load-dir', default='./marlsave', help='directory to load all policies from')
    parser.add_argument('--swarm-load-run', default=None, help='run to load all policies from, {a number or "latest"')
    parser.add_argument('--swarm-load-ckpt', default=None, help='checkpoint to load from, {a number or "latest"}')
    parser.add_argument('--adversary-load-path', default=None, help='path to load adversary model from')
    parser.add_argument('--adversary-load-dir', default='./adversary_training/runs', help='directory to load adversary model from')
    parser.add_argument('--adversary-load-run', default=None, help='run to load all policies from, {a number or "latest"')
    parser.add_argument('--adversary-load-ckpt', default=None, help='checkpoint to load from, {a number or "latest"}')
    

    # we always set these to TRUE, so automating this
    parser.add_argument('--no-clipped-value-loss', action='store_true')
    
    args = parser.parse_args()



    args.clipped_value_loss = not args.no_clipped_value_loss

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda:{}".format(args.cuda_id) if args.cuda else "cpu")
    args.train_adversary = args.train_adversary and args.use_adversary
    args.num_processes = 1 if args.algo=='genetic' else args.num_processes

    # -------- buffer match to nearest divisor -------- #
    # args.num_episodes_iter = 
    args.num_steps_buffer = args.buffer_size//args.num_processes
    args.buffer_size = args.num_steps_buffer*args.num_processes
    if args.adversary_num_trajs == None:
        args.adversary_num_trajs = args.buffer_size//args.num_steps_episode
    args.continue_from_iter = 0
    if args.algo == 'ppo':
        args.update_every = args.buffer_size
        args.batch_size = args.buffer_size

    elif args.algo == 'sac':
        args.num_updates = args.update_every


    # -------- random leader name only for testing -------- #
    if args.mode == 'train':
        args.random_leader_name = 0


    # -------- load paths for pretrained swarm and adversary -------- #
    if args.load_mode == 'individual':
        if args.swarm_load_path is None:
            args.swarm_load_run = process_runID(args.swarm_load_run, args.swarm_load_dir)
            args.swarm_load_ckpt = process_ckpt(args.swarm_load_ckpt, args.swarm_load_dir, args.swarm_load_run)
            print('processed', args.swarm_load_ckpt, 'joint_argumenst.pst')
            if not args.swarm_load_run is None:
                args.swarm_load_path = os.path.join(args.swarm_load_dir, 'run_'+str(args.swarm_load_run), 'ckpt_'+str(args.swarm_load_ckpt)+'.pt')

        if args.adversary_load_path is None:
            args.adversary_load_run = process_runID(args.adversary_load_run, args.adversary_load_dir)
            args.adversary_load_ckpt = process_ckpt(args.adversary_load_ckpt, args.adversary_load_dir, args.adversary_load_run)
            if not args.adversary_load_run is None:
                args.adversary_load_path = os.path.join(args.adversary_load_dir, 'run_'+str(args.adversary_load_run), 'ckpt_'+str(args.adversary_load_ckpt)+'.pt')

    elif args.load_mode == 'joint':
        if args.load_path is None:
            args.load_run = process_runID(args.load_run, args.root_save_dir)
            args.load_ckpt = process_ckpt(args.load_ckpt, args.root_save_dir, args.load_run)
            if not args.load_run is None:
                args.load_path = os.path.join(args.root_save_dir, 'run_'+str(args.load_run), 'ckpt_{}.pt'.format(args.load_ckpt))

    if args.continue_training:
        runID = args.load_run

        num_frames = args.load_ckpt*args.save_interval
        args.continue_from_iter = num_frames // args.num_steps_episode // args.num_processes
        
        args.data_temp_dir = os.path.join(args.data_temp_dir, str(runID))
        args.save_dir = os.path.join(args.root_save_dir,'run_{}'.format(runID))
        args.log_dir = os.path.join(args.save_dir, args.log_dir)

    # -------- create directory if starting to train -------- #
    if not (args.continue_training or args.mode == 'test'):
        rootDir = args.root_save_dir
        if not os.path.exists(rootDir):
            os.makedirs(rootDir)
            runID = 0
        else:
            runs = [int(d.split('_')[-1]) for d in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir,d))]
            runID = max(runs)+1 if len(runs)>0 else 0

        args.data_temp_dir = os.path.join(args.data_temp_dir, str(runID))
        args.save_dir = os.path.join(rootDir,'run_{}'.format(runID))
        os.makedirs(args.save_dir)
        
        
        args.log_dir = os.path.join(args.save_dir, args.log_dir)

    # -------- load if continue training or test. load-dir = ckpt file path -------- #
    # else:
    #     assert args.algo=='scripted' or args.load_mode == 'individual' or args.load_mode == 'joint'
    #     if args.load_mode == 'joint':
    #         assert (not args.load_path is None) or not (args.load_run is None or args.load_ckpt is None), 'please provide load-path or (load-ckpt and load-run)'
            
    #         if args.load_path is None:
    #             if args.load_run == 'latest':
    #                 rootDir = args.root_save_dir
    #                 args.load_run = max([int(d.split('_')[-1]) for d in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir,d))])
    #             args.load_run = int(float(args.load_run))

    #             if args.load_ckpt == 'latest':
    #                 run_dir = os.path.join(args.root_save_dir, 'run_'+str(args.load_run))    
    #                 args.load_ckpt = max([int(file[5:][:-3]) for file in os.listdir(run_dir) if file.endswith('.pt')])
    #             args.load_ckpt = int(float(args.load_ckpt))

    #             if args.continue_training:
    #                 runID = args.load_run
    #                 args.continue_from_iter = args.load_ckpt*args.save_interval
    #                 args.data_temp_dir = os.path.join(args.data_temp_dir, str(runID))
    #                 args.save_dir = os.path.join(rootDir,'run_{}'.format(runID))

    #             args.load_path = os.path.join(args.root_save_dir, 'run_'+str(args.load_run), 'ckpt_'+str(args.load_ckpt)+'.pt')

    if args.continue_training:
        assert args.load_path is not None and os.path.exists(args.load_path), \
        "Please specify valid model file to load if you want to continue training"

    if args.identity_size > 0:
        assert args.identity_size >= args.num_agents, 'identity size should either be 0 or >= number of agents!'

    # -------- output traj and video directory -------- #
    if args.mode == 'test':
        if not args.out_dir is None:
            args.out_dir = os.path.join('output', args.out_dir)
        else:
            args.out_dir = os.path.join('output', 'num_agents_{}'.format(args.num_agents))
            # args.store_video_together = True

        if args.record_video:
            if args.store_video_together:
                args.video_path = os.path.join('output', 'videos')
            else:
                args.video_path = os.path.join(args.out_dir, 'videos')
            if not os.path.exists(args.video_path):
                os.makedirs(args.video_path)
   

    if not args.masking:
        args.mask_dist = None
    elif args.masking and args.dropout_masking:
        args.mask_dist = -10

    
    return args
