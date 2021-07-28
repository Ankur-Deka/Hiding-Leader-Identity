# -------- import GridSearch and define/import the compile function -------- #
import sys
sys.path.append('./SIGS-Grid-Search')
from grid_search import GridSearch

# -------- main file to run -------- #
# mode = 'train'
mode = 'test'
num_process = 3



main_file = 'joint_main_genetic.py'
# 'gen_plot_data_genetic.py'#'joint_main_genetic.py'#'cross_evaluation.py'
if mode == 'train':
	args = {'mode': ['train'],
			'algo': ['genetic'],
			'env-name': ['simple_flocking'],
			# 'use-adversary': [0],
			# 'adversary-num-trajs': [100],
			'adversary-version': ['V2'],
			'adversary-hidden-dim': [512],
			'adversary-num-epochs': [1],
			'adversary-genetic-beta': [0.5],#[0.2, 0.5, 0.8],
			# 'num-steps-episode': [100],
			# 'continue-training': [''],
			# 'train-adversary': [0],
			# 'num-agents': [5],
			# 'privacy-reward': [1,2],#,0.1,0.5,1,2,5],
			# 'seed': [100,101],
			'num-frames': [1000000],
			# 'lr': [1e-3],
			'num-processes': [1],
			'render': [''],
			# 'load-mode': ['joint'],
			# 'load-run': [100], #{'min':20, 'max':17, 'num': 1, 'scale': 'linear'},
			# 'load-ckpt': ['latest'],#,10,20,30,40,50,60,70,80,90,100],
			'load-mode': ['individual'],
			# 'swarm-load-path': ['./marlsave_privacy/run_32/ckpt_100.pt'],
			# 'swarm-load-run': [None, 32, 0],
			# 'swarm-load-ckpt': ['latest'],
			'adversary-load-run': [8],
			'adversary-load-ckpt': ['latest']
			}

# python joint_main.py --mode test --use-adversary 0 --load-mode joint --load-run 0 --load-ckpt latest --out-dir stage_1_test --num-eval-episodes 100 --goal-at-top

elif mode == 'test':
	args = {'mode': ['test'],
			'env-name': ['simple_flocking'],#['simple_flocking'],
			'algo': ['genetic'], # 'scripted, genetic, genetic_random, ppo'
			'adversary-version': ['V2'],
			'adversary-hidden-dim': ['512'],
			# 'privacy-reward': [2],
			'load-mode': ['joint'],
			'load-run': [229], #{'min':20, 'max':17, 'num': 1, 'scale': 'linear'},
			'load-ckpt': ['latest'],#,10,20,30,40,50,60,70,80,90,100],
			# 'random-leader-name': [1],
			# 'load-ckpt': ['latest'],
			'num-eval-episodes': [10],
			# 'num-cross-eval-ckpts': [100],
			'seed': [0],
			'record': [''],
			'goal-at-top': [''],
			'plot-trajectories': [''],
			# 'store-video-together': [''],
			# 'video-format': ['webm'],
			'render': [''],
			
			# 'use-adversary': [0],
			# 'num-steps-episode': [50],
			'algo-stage': ['genetic'],
			'same-color': [''],
			# 'hide-goal': [''],
			
			'out-dir': ['tmp'],
			# 'out-dir': ['tmp_dataset'],
			# 'swarm-load-dir': ['./marlsave_privacy'],
			# 'swarm-load-run': [35],
			# 'swarm-load-ckpt': ['latest'],
			# # 'adversary-load-path': ['./adversary_training/runs/run_26/ckpt_99.pt'],
			
			# 'adversary-load-dir': ['./adversary_training/runs'],
			# 'adversary-load-run': [1],
			# 'adversary-load-ckpt': ['latest']
			}

# -------- create GridSearch object and run -------- #
myGridSearch = GridSearch(main_file, args, num_process=num_process)
myGridSearch.run()
