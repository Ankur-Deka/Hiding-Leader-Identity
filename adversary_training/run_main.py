# -------- import GridSearch and define/import the compile function -------- #
import sys
sys.path.append('../SIGS-Grid-Search')
from grid_search import GridSearch

# -------- main file to run -------- #
main_file = 'main.py'

# python main.py --mode test --version V1 --loadRun 26 --visualize --initSteps 1
# python main.py --mode train --initSteps 1
# from grid_search import GridSearch


# -------- define dictionary of arguments for grid search -------- #
# args = {'mode': ['train'],
# 		'version': ['V0', 'V1', 'V2'],
# 		# 'dataDir': ['../trajectory_datasets/dataset_2']
# 		'lr': [1e-4, 5e-4, 1e-3]
# 		# 'hiddenDim': [32]
# 	   }

args = {'mode': ['test'],
		'version': ['V2'],
		'initSteps': [1],
		'loadRun': [23],
		'loadCkpt': [199],
		'visualize': ['']
}


# args = {'mode': ['test'],
# 		'version': ['V1'],
# 		# 'initSteps': [1],
# 		'loadRun': [10],
# 		'loadCkpt': [199],
# 		'visualize': ['']
# }
# -------- create GridSearch object and run -------- #
myGridSearch = GridSearch(main_file, args, num_process=9)
myGridSearch.run()
