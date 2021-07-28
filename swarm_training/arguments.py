import argparse
import os
import sys
import torch
import shutil


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    
    # environment
    parser.add_argument('--env-name', default='simple_flocking', help='one from {simple_spread, simple_formation, simple_line, simple_flocking}, first 3 may not work)')
    parser.add_argument('--num-agents', type=int, default=3)
    parser.add_argument('--masking', action='store_true', help='restrict communication to within some threshold')
    parser.add_argument('--mask-dist', type=float, default=1.0, help='distance to restrict comms')
    parser.add_argument('--dropout-masking', action='store_true', help='dropout masking enabled')
    parser.add_argument('--entity-mp', action='store_true', help='enable entity message passing')
    parser.add_argument('--identity-size', default=0, type=int, help='size of identity vector')
    parser.add_argument('--num-steps-episode', type=int, default=50)
    parser.add_argument('--vec-normalize', action = 'store_true')    

    # training 
    parser.add_argument('--seed', type=int, default=None, help='random seed (default: None)')
    parser.add_argument('--num-processes', type=int, default=32, help='how many training CPU processes to use (default: 32)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--num-frames', type=int, default=int(5e5), help='number of frames to train (default: 50e6)')
    parser.add_argument('--arena-size', type=int, default=1, help='size of arena')
    
    # evaluation
    parser.add_argument('--num-eval-episodes', type=int, default=10, help='number of episodes to evaluate with')
    parser.add_argument('--dist-threshold', type=float, default=0.1, help='distance within landmark is considered covered (for simple_spread)')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--record-video', action='store_true', default=False, help='record evaluation video')
    parser.add_argument('--store-traj', action='store_true', )

    # PPO/SAC
    parser.add_argument('--algo', default='ppo', help='algorithm to use: a2c | ppo | acktr | sac')
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
    parser.add_argument('--update-every', type=int, default=50, help='update after this much interval, set automatically for PPO')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size sampled from buffer for SAC iterations. automatically set for PPO')


    # logging
    parser.add_argument('--root-save-dir', default='./marlsave', help='Root directory to save models. Relative path is auto generated')
    parser.add_argument('--log-dir', default='logs', help='directory to save logs')
    parser.add_argument('--save-interval', type=int, default=10000, help='save interval in terms of env steps')
    parser.add_argument('--log-interval', type=int, default=10, help='log interval, one log per n updates (default: 10)')
    
    # Miscellaneous
    parser.add_argument('--test', action='store_true')
    # -------- at test, provide either load-dir or load-run & load-ckpt. load-dir is considered if all provided -------- #
    parser.add_argument('--load-dir', default=None, help='filename to load all policies from')
    parser.add_argument('--load-run', default=None, help='run to load all policies from, {a number or "latest"')
    parser.add_argument('--load-ckpt', default=None, help='checkpoint to load from, {a number or "latest"}')
    parser.add_argument('--eval-interval', default=10, type=int)
    parser.add_argument('--continue-training', action='store_true')
    parser.add_argument('--diff-reward', type=int, default=1, help='{0,1} Whether to use difference in distance reward for swarm?')
    parser.add_argument('--same-color', action='store_true')
    parser.add_argument('--goal-at-top', action='store_true')
    parser.add_argument('--hide-goal', action='store_true')
    parser.add_argument('--video-format', default='webm')
    parser.add_argument('--random-leader-name', action='store_true', help='During evaluation, assigns a random name to leader. This is just a name, in agents list first agent is always the leader')
    parser.add_argument('--out-dir', default=None, type=str, help='Which folder do you wish to store the trajectories and videos in?. Final path is "output/out-dir"')
    parser.add_argument('--store-video-together', action='store_true', help='if passed, all videos are saved in "output/videos". Else videos are saved in "output/out-dir/videos"')

    # we always set these to TRUE, so automating this
    parser.add_argument('--no-clipped-value-loss', action='store_true')
    
    args = parser.parse_args()

    args.clipped_value_loss = not args.no_clipped_value_loss

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    
    # -------- buffer match to nearest divisor -------- #
    args.num_steps_buffer = args.buffer_size//args.num_processes
    args.buffer_size = args.num_steps_buffer*args.num_processes

    if args.algo == 'ppo':
        args.update_every = args.buffer_size
        args.batch_size = args.buffer_size

    elif args.algo == 'sac':
        args.num_updates = args.update_every


    # -------- random leader name only for testing -------- #
    if not args.test:
        args.random_leader_name = False

    # -------- create directory if starting to train -------- #
    if not (args.continue_training or args.test):
        rootDir = args.root_save_dir
        if not os.path.exists(rootDir):
            os.makedirs(rootDir)
            runID = 0
        else:
            runs = [int(d.split('_')[-1]) for d in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir,d))]
            runID = max(runs)+1 if len(runs)>0 else 0

        args.save_dir = os.path.join(rootDir,'run_{}'.format(runID))
        os.makedirs(args.save_dir)
        
        args.log_dir = args.save_dir + '/' + args.log_dir

    # -------- load if continue training or test. load-dir = ckpt file path -------- #
    else:
        assert (not args.load_dir is None) or not (args.load_run is None or args.load_ckpt is None) 
        if args.load_dir is None:
            if args.load_run == 'latest':
                rootDir = args.root_save_dir
                args.load_run = max([int(d.split('_')[-1]) for d in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir,d))])
            args.load_run = int(float(args.load_run))

            if args.load_ckpt == 'latest':
                run_dir = os.path.join(args.root_save_dir, 'run_'+str(args.load_run))    
                args.load_ckpt = max([int(file[5:][:-3]) for file in os.listdir(run_dir) if file.endswith('.pt')])
            args.load_ckpt = int(float(args.load_ckpt))

            args.load_dir = os.path.join(args.root_save_dir, 'run_'+str(args.load_run), 'ckpt_'+str(args.load_ckpt)+'.pt')

    if args.continue_training:
        assert args.load_dir is not None and os.path.exists(args.load_dir), \
        "Please specify valid model file to load if you want to continue training"

    if args.identity_size > 0:
        assert args.identity_size >= args.num_agents, 'identity size should either be 0 or >= number of agents!'

    # -------- output traj and video directory -------- #
    if args.test:
        if not args.out_dir is None:
            args.out_dir = os.path.join('output', args.out_dir)
            path = os.path.join(args.out_dir, 'traj')
            if not os.path.exists(path):
                os.makedirs(path)
        else:
            args.store_video_together = True

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
    

    # raise warning if save directory already exists
    # if not args.test:
    #     if os.path.exists(args.save_dir):
    #         print('\nSave directory exists already! Enter')
    #         ch = input('c (rename the existing directory with _old and continue)\ns (stop)!\ndel (delete existing dir): ')
    #         if ch == 's':
    #             sys.exit(0)
    #         elif ch == 'c':
    #             os.rename(args.save_dir, args.save_dir+'_old')
    #         elif ch == 'del':
    #             shutil.rmtree(args.save_dir)
    #         else:
    #             raise NotImplementedError('Unknown input')
    #     os.makedirs(args.save_dir)
    
    return args
