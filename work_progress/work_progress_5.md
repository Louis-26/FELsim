2026/06/03

# tasks
- ✅produce the plot of (alpha_x,alpha_y) scatter plot for each method
- ✅produce the heat map/contour plot of MSE landscape, and show the iteration trajectory of each method on the landscape 
- ✅adjust the method to improve the L-BFGS-B/SLSQP converged value from torch float64
- ✅consolidate all methods into .py file, only call the functions in the .ipynb file, 
- ✅increase the trial number from 100 to 1000, and summarize the results
- ✅reproduce the results with numpy version(Prof Bidault original code in paper), and compare the results with the pytorch version



# finished work
- based on the scatter plot of the converged alpha_x, alpha_y values, we can find out that **L-BFGS-B** gives much better convergence results than any other method with superior stability, 
- in the region of (0.01, 1.5) for I_1 and I_3, three local minima exist, and each method would converge to various local minima, meaning that deeper analysis is needed
- Converting to torch float64 gives much better model performance in terms of L-BFGS-B and trust-constr, but Nelder-Mead strangely performs worse
- intermediate results have been saved in .pkl files under the folder `results`
- code has been consolidated into [experiments_utils.py](../experiment/experiments_utils.py)
- In order for the scatter plot to have better generalizability and representativeness, the trial number has been increased to 1000, taking total time of **23 hours 53 minutes**
- It turns out that the results from numpy version and pytorch version are very similar, but the reproduced one has some discrepancy with the original paper

# potential next step
1. Consider hyperparameter tuning for better model stability by cross validation.
2. Step into scenario B and C with selected algorithms as model benchmark.
3. Use Koa cluster for parallelization to speed up the experiment.