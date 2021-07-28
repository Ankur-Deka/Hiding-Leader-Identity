# -------- import GridSearch and define/import the compile function -------- #
import sys
sys.path.append('../SIGS-Grid-Search')
from grid_search import GridSearch

# -------- main file to run -------- #
mode = 'train' # train or test
num_process = 4

if mode == 'train':
	main_file = 'main.py'

	args = {'env-name': ['simple_flocking'],
			'num-agents': [6],
			'algo':['ppo'],
			'buffer-size':[500],
			'eval-interval':[2000],
			'num-updates': [4],
			'num-frames': [500000],
			'lr': {'min':-3, 'max':-2, 'num':1, 'scale': 'log'},
			'num-processes': [1],
			'render': ['']}

elif mode == 'test':
	main_file = 'eval.py'

	args = {'test': [''],
			'env-name': ['simple_flocking'],
			'num-agents': [6],
			'algo':['ppo'],
			'load-run': [1],#{'min':163, 'max':165, 'num': 2, 'scale': 'linear'},
			'load-ckpt': ['latest'],
			'num-eval-episodes': [100],
			# 'seed': [1492],
			# 'record': [''],
			# 'same-color': [''],
			# 'render': [''],
			# 'random-leader-name': [''],
			# 'out-dir': ['test_dataset']
			}

# -------- create GridSearch object and run -------- #
myGridSearch = GridSearch(main_file, args, num_process=num_process)
myGridSearch.run()