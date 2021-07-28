# Intro
This script will help you run grid search on any python script. 

# Dependencies
`multiprocessing`, `numpy`, `csv`

# File structure
Your code should include the following elements:
1. A main file which accepts arguments with argparse. Let's call is `main.py`.
1. A function to compile the data returned by `main.py`, let's call it `compile_func`. This function should be able to take the data points iteratively. You'd have to call this from `main.py`. A simple function called is already provided and can be imported as `from grid_search import insert_to_csv`.

# Instructions
It it best to open `dummy_run.py` as the library is quite intuitive and simple to run. You can use this file as your template.

1. Import the library `from grid_search import GridSearch`
1. Set name of main file, `main_file = 'dummy_main.py'`
1. Create a function for compiling result. I am importing the `compile_csv` function from `grid_search`. I am calling this from within my `dummy_main.py` file which saves the result of every run in a csv file.
1. Define arguments for running grid search:
	1. For every parameter you'd need one key containing it's name
	1. Each key can either be a list of all values of that parameter
	1. Or each key can be a dictionary with keys {'min', 'max', 'num', 'scale'} if you wish to uniformly choose values in a range. 'scale' can be 'linear' or 'log'. For log scale 'min' and 'max' should be the powers of 10. Eg. for 1e-1 to 1e3, 'min'=-1 and 'max'=3 
1. Create an object of `GridSearch`. `myGridSearch = GridSearch(main_file, compile_func, args, num_process=2)`
1. Run grid search with `myGridSearch.run()` 

