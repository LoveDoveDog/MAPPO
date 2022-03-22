import os
from datetime import datetime
import torch
import numpy as np

from module_ppo import PPO
from module_env import bigenv
from module_others import new_arg, state_decompose

# In this program, you need only to modify the specialized hyperparmeters part

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")
################ PPO hyperparameters ################
episode_timestep_length = 1000  # max steps in one episode
total_timestep_length = int(3e6)  # maximum training steps
log_episode_interval = 2
log_timestep_interval = episode_timestep_length * log_episode_interval
printout_episode_interval = 10
printout_timestep_interval = episode_timestep_length * printout_episode_interval
save_timestep_interval = printout_timestep_interval * 5  # save model frequency
update_timestep_interval = episode_timestep_length * 4  # update policy
K_epochs = 80           # update policy for K epochs in one PPO update
eps_clip = 0.2          # clip parameter for PPO
gamma = 0.99            # discount factor
lr_actor = 0.0003       # learning rate for actor network
lr_critic = 0.001       # learning rate for critic network
####### initialize environment hyperparameters ######
random_seed = 0
torch.manual_seed(random_seed)
np.random.seed(random_seed)
args = new_arg()
############ Specialized hyperparameters ############
args.actor_node=32
args.critic_node=32
args.agent_num=3
################# save setting file #################
env_dir = "PPO_alg" + '/'
if not os.path.exists(env_dir):
    os.makedirs(env_dir)
env_num = len(next(os.walk(env_dir))[2])
env_name = env_dir + 'env_' + str(env_num) + '/'
if not os.path.exists(env_name):
    os.makedirs(env_name)
set_file_handler = open(env_dir+'env_'+str(env_num)+'.csv', 'w+')
set_file_handler.write('actor_node,{}\n'.format(args.actor_node))
set_file_handler.write('critic_node,{}\n'.format(args.critic_node))

set_file_handler.close()
###################### logging ######################
log_path_and_name = env_name + "log.csv"
print("Data logging at : " + log_path_and_name)
################### model-saving ####################
print("Model saving at : " + env_name)
################ training procedure #################
env = bigenv(args)
# initialize PPO agents
for index_for_large in np.arange(args.agent_num):
    locals()['ppo_agent_'+str(index_for_large).zfill(3)] = PPO(env.state_dim_critic,
                                                               env.state_dim_actor, env.action_dim, args.actor_node, args.critic_node, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device)
# ppo_agent=PPO(3,2,lr_actor, lr_critic, gamma, K_epochs, eps_clip)
start_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)
print("============================================================================================")
# logging file
log_handler = open(log_path_and_name, "w+")
log_handler.write('episode,timestep,reward\n')
# printing and logging variables
log_sum_reward = np.zeros(args.agent_num+1)
printout_sum_reward = np.zeros(args.agent_num+1)
episode_counter = 0
timestep_counter = 0
current_state = env.reset()
while timestep_counter <= total_timestep_length:  
    current_episode_reward = np.zeros(args.agent_num+1)
    for _ in np.arange(episode_timestep_length):
        timestep_counter += 1
        current_action = []
        state_for_common = torch.FloatTensor(current_state).to(device)
        for index_for_large in np.arange(args.agent_num):
            state_for_ppo =  state_decompose(state_for_common, index_for_large)
            action_for_ppo = locals()[
                'ppo_agent_'+str(index_for_large).zfill(3)].select_action(state_for_ppo)
            current_action.append(action_for_ppo)
        current_action=np.array(current_action)
        current_state, reward_list = env.step(current_action)
        for index_for_large in np.arange(args.agent_num):
            locals()['ppo_agent_'+str(index_for_large).zfill(3)
                     ].buffer.rewards.append(reward_list[index_for_large])
            locals()['ppo_agent_'+str(index_for_large).zfill(3)
                     ].buffer.states_for_common.append(state_for_common)
        current_episode_reward += np.array(reward_list)
        # update PPO agent
        if timestep_counter % update_timestep_interval == 0:
            for index_for_large in np.arange(args.agent_num):
                locals()['ppo_agent_'+str(index_for_large).zfill(3)].update()
        # log file
        if timestep_counter % log_timestep_interval == 0:
            log_avg_reward = log_sum_reward / log_timestep_interval
            log_avg_reward = np.around(log_avg_reward, 4) 
            log_handler.write('{},{},{}\n'.format(
                str(episode_counter-log_episode_interval+2)+'-'+str(episode_counter+1), timestep_counter, log_avg_reward))
            log_handler.flush()
            log_sum_reward = np.zeros(args.agent_num+1)
        # print average reward
        if timestep_counter % printout_timestep_interval == 0:
            print_avg_reward = printout_sum_reward / printout_timestep_interval
            print_avg_reward = np.around(print_avg_reward, 2)
            print("Episode : {}-{} \t Timestep : {} \t Average Reward : {}".format(
                episode_counter-printout_episode_interval+2, episode_counter+1, timestep_counter, print_avg_reward))
            printout_sum_reward = np.zeros(args.agent_num+1)
        # save model weights
        if timestep_counter % save_timestep_interval == 0:
            print(
                "--------------------------------------------------------------------------------------------")
            print("saving model at : " + env_name)
            for index_for_large in np.arange(args.agent_num):
                model_path_and_name = env_name + \
                    "model_" + str(index_for_large) + ".pth"
                locals()['ppo_agent_'+str(index_for_large).zfill(3)
                         ].save(model_path_and_name)
            print("model saved")
            print("Elapsed Time  : ", datetime.now().replace(
                microsecond=0) - start_time)
            print(
                "--------------------------------------------------------------------------------------------")

    episode_counter += 1
    log_sum_reward += current_episode_reward
    printout_sum_reward += current_episode_reward

log_handler.close()

# print total training time
print("============================================================================================")
end_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)
print("Finished training at (GMT) : ", end_time)
print("Total training time  : ", end_time - start_time)
print("============================================================================================")