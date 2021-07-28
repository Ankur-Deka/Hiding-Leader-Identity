# -------- import GridSearch and define/import the compile function -------- #
import sys
sys.path.append('./SIGS-Grid-Search')
from grid_search import GridSearch

# -------- main file to run -------- #
# mode = 'train'
mode = 'test'
num_process = 4



main_file = 'joint_main.py'#'cross_evaluation.py'
if mode == 'train':
	args = {'mode': ['train'],
			'env-name': ['simple_flocking'],
			'cuda-id': [3],
			# 'algo': ['genetic'],
			# 'use-adversary': [0],
			# 'adversary-num-trajs': [100],
			# 'adversary-version': ['V2'],
			# 'adversary-hidden-dim': [512],
			# 'adversary-num-epochs': [1],
			# 'num-steps-episode': [100],
			# 'continue-training': [''],
			'train-swarm': [0],
			# 'train-adversary': [0],
			# 'num-agents': [5],
			# 'privacy-reward': [0.2],#,0.1,0.5,1,2,5],
			# 'seed': [100,101],
			'num-frames': [1000000],
			# 'lr': [1e-3],
			'save-interval': [500],
			'num-processes': [10],
			# 'render': [''],
			'load-mode': ['joint'],
			'load-run': [1], #{'min':20, 'max':17, 'num': 1, 'scale': 'linear'},
			'load-ckpt': ['latest'],#,10,20,30,40,50,60,70,80,90,100],
			# 'load-mode': ['individual'],
			# 'swarm-load-path': ['./marlsave_privacy/run_32/ckpt_100.pt'],
			# 'swarm-load-run': [None],
			# 'swarm-load-ckpt': ['latest'],
			# 'adversary-load-run': [0],
			# 'adversary-load-ckpt': ['latest']
			}

# python joint_main.py --mode test --use-adversary 0 --load-mode joint --load-run 0 --load-ckpt latest --out-dir stage_1_test --num-eval-episodes 100 --goal-at-top

elif mode == 'test':
	args = {'mode': ['test'],
			'env-name': ['simple_flocking'],#['simple_flocking'],
			'num-agents': [6],#,4,5,6,7,8,9,10,11,12],
			# 'algo': ['scripted'], # 'scripted, genetic, genetic_random, ppo'
			# 'adversary-version': ['V2'],
			# 'adversary-hidden-dim': ['512'],
			# 'privacy-reward': [2],
			'load-mode': ['joint'],# 'individual'],
			'load-run': [256], #{'min':20, 'max':17, 'num': 1, 'scale': 'linear'},
			# 'load-ckpt': [1,250,500,750,1000,1250,1500,1750,2000],
			'load-ckpt': [1,20,30,40,50,60,70,80,90,100],
			# 'random-leader-name': [1],
			# 'load-ckpt': ['latest'],
			'num-eval-episodes': [3],
			# 'num-cross-eval-ckpts': [100],
			'seed': [0],
			'record': [''],
			# 'goal-at-top': [''],
			'plot-trajectories': [''],
			'store-video-together': [''],
			# 'video-format': ['webm'],
			# 'render': [''],
			
			# 'use-adversary': [0],
			# 'num-steps-episode': [50],
			# 'algo-stage': [4.3],
			# 'same-color': [''],
			# 'hide-goal': [''],
			
			# 'out-dir': ['tmp'],
			# 'out-dir': ['tmp_dataset'],
			# 'swarm-load-dir': ['./marlsave_privacy'],
			# 'swarm-load-run': [0],
			# 'swarm-load-ckpt': ['latest'],
			# # 'adversary-load-path': ['./adversary_training/runs/run_26/ckpt_99.pt'],
			
			# 'adversary-load-dir': ['./adversary_training/runs'],
			# 'adversary-load-run': [0],
			# 'adversary-load-ckpt': ['latest']
			}

# -------- create GridSearch object and run -------- #
myGridSearch = GridSearch(main_file, args, num_process=num_process)
myGridSearch.run()
