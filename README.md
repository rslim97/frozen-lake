# Deep-learning-module-project
Project 1
- Solve Frozen Lake environment from OpenAI gym using Monte Carlo control method and Temporal Difference learning methods such as Q-learning and SARSA. 

Training codes for MC control: 
mc.py - for original 4x4 grid problem, mc_extended.py- grid extended to 10x10.

Training codes for Q-learning: 
Q_learning.py - for original 4x4 grid problem, Q_learning_extended.py- grid extended to 10x10.

Training codes for SARSA: 
sarsa.py - for original 4x4 grid problem, sarsa_extended.py- grid extended to 10x10.

main.py - put all the codes in a one folder together with this main.py file, run 'python main.py' in Anaconda Prompt to execute everything.
Note: the codes were run using Python 3.6.4 version

Jupyter notebook for figures in report:
frozenlake_auxiliary.ipynb - the contents are messy but it contains original code to generate the plots in the report.
Note: May take long time to finish running the entire notebook and the resulting plot may be different because of the stochastic nature of the algorithms.

Python libraries that need to be installed- gym and matplotlib for frozenlake_auxiliary.ipynb
