import argparse
import numpy as np

# You can add more variable definitions in def new_arg() 
# and then change their values in module_train.py

def new_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--actor_node', default=64, type=int)
    parser.add_argument('--critic_node', default=128, type=int)
    parser.add_argument('--agent_num', default=3, type=int)
    args=parser.parse_args()
    return args


def state_decompose(state_for_common, index):
    state_for_ppo = [state_for_common[index]]
    state_for_ppo = np.array(state_for_ppo)
    return state_for_ppo # has to be a horizontial numpy vector