# MAPPO
A module for multi-agent PPO (MAPPO) and is pretty friendly to the applications in IEEE field.

The programs refer the PPO programs in https://github.com/nikhilbarhate99/PPO-PyTorch. Currently, this program only works for discrete-valued action cases.

To use this module, you need to first write your own environment program in module_env.py, where a sample environment is given. Next, you need to specific the state decomposition function in module_others.py, which is to decompose the common state to individual observations. Then, you can add some key variables in the parser of module_others.py (optional). Finally, you can train your own MAPPO by executing module_train.py. 

Do not modify module_ppo.py. And do use the correct data format when constructing your own state, observation, etc. I have left comments at these positions.
