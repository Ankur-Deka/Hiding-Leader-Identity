# -------- import GridSearch and define/import the compile function -------- #
from grid_search import GridSearch

# -------- main file to run -------- #
main_file = 'dummy_main.py'

# -------- define dictionary of arguments for grid search -------- #
args = {'num1': [2,5,6],
		'num2': {'min':1, 'max':2, 'num': 5, 'scale': 'linear'}}

# -------- create GridSearch object and run -------- #
myGridSearch = GridSearch(main_file, args, num_process=2)
myGridSearch.run()