import gym
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import uniform_filter1d
from typing import Type
import logging

from algorithms.semi_grad_sarsa import semi_grad_sarsa
from algorithms.reinforce import reinforce_with_baseline
from algorithms.actor_critic import one_step_actor_critic
from utils.helpers import Action, StateRepresentation, Parametrized
from utils.containers import Fourier_SAR, Fourier_SR, FourierInd_SAR, FourierInd_SR, QFunc, ValueFunc, PolicyFunc

save_params = ['num_steps', 'rewards']

def demo(env, q_func:QFunc, SAR, SAR_args):
    observation, info = env.reset()
    for _ in range(400):
        action = q_func.greedy(SAR, SAR_args, StateRepresentation(observation))
        ac = action_mapping(action)
        observation, reward, terminated, truncated, info = env.step(ac)
        if terminated or truncated:
            observation, info = env.reset()
    env.close()

def demo_new(env, policy_func:PolicyFunc, SAR, SAR_args):
    observation, info = env.reset()
    for _ in range(400):
        action = policy_func.sample(SAR, SAR_args, StateRepresentation(observation), 1.0)

        ac = action_mapping(action)

        observation, reward, terminated, truncated, info = env.step(ac)
        if terminated or truncated:
            observation, info = env.reset()
    env.close()

def save_func_data(*arrays, data:dict=None, save_loc:str='data.npz')->None:
    with open(save_loc, 'wb') as f:
        arrays = {f'array_{i}': arrays[i] for i in range(len(arrays))}
        if data is not None:
            for arr_name in save_params:
                try:
                    arrays[arr_name] = data[arr_name]
                except KeyError:
                    logging.warning(f'Could not save: {arr_name}')
        np.savez(f, **arrays)

def load_func(n:int, save_loc:str)->tuple[list, dict]:
    arrays = []
    data = dict()
    with open(save_loc, 'rb') as f:
        npz_file = np.load(f)
        for i in range(n):
            arrays.append(npz_file[f'array_{i}'])
        for arr_name in save_params:
            try:
                data[arr_name] = npz_file[arr_name]
            except KeyError:
                logging.warning(f'Could not load: {arr_name}')
    
    return arrays, data

def plot_steps(num_steps:np.ndarray, N=1, title=None):
    y = uniform_filter1d(num_steps, size=N)
    fig, ax = plt.subplots()
    ax.plot(y)
    if title == 'steps':
        ax.set_title('Steps V/S Episodes')
        plt.xlabel("No. of Episodes")
        plt.ylabel("No. of Steps")
    elif title == 'rewards':
        ax.set_title('Reward V/S Episodes')
        plt.xlabel("No. of Episodes")
        plt.ylabel("Total Reward")
    plt.show()

def plot_steps_v2(all_results:dict, variable:str, N=1, title=None):
    for val, data in all_results.items():
        y = uniform_filter1d(data, size=N)
        plt.plot(y, label= f"{variable}={val}")
    if title == 'steps':
        plt.title('Steps V/S Episodes')
        plt.xlabel("No. of Episodes")
        plt.ylabel("No. of Steps")
    elif title == 'rewards':
        plt.title('Reward V/S Episodes')
        plt.xlabel("No. of Episodes")
        plt.ylabel("Total Reward")
    plt.legend(loc="upper right")
    plt.show()

def plot_steps_v3(actions_list: list, steps_list: list):
    #take averages
    total_actions = np.average(actions_list, axis=0)
    total_steps = np.average(steps_list, axis=0)
    std_dev = np.std(steps_list, axis=0)
    total_iterations = range(len(total_steps))

    plt.plot(total_actions, total_iterations)
    plt.ylabel("No. of episodes completed")
    plt.xlabel("No. of actions taken")
    plt.title("Episodes VS Actions")
    plt.show()

    plt.errorbar(total_iterations, total_steps, std_dev, linestyle='-', ecolor='lightsteelblue')
    plt.ylabel("No. of steps")
    plt.xlabel("No. of episodes completed")
    plt.title("Steps VS Episodes")
    plt.show()

def recons(c_theta, s_theta):
    if c_theta>=0:
        return np.arcsin(s_theta)
    else:
        return np.arccos(c_theta) if s_theta>=0 else -np.arccos(c_theta)

def acrobot_theta_recons(observation:np.ndarray)->np.ndarray:
    ct1, st1 = observation[0], observation[1]
    ct2, st2 = observation[2], observation[3]
    theta1, theta2 = recons(ct1, st1), recons(ct2, st2)

    return np.array([theta1, theta2, observation[4], observation[5]])

def action_mapping(action:Action)->int:
    if np.isclose(action.action_num, 0.1):
        ac = 0
    elif np.isclose(action.action_num, 0.17):
        ac = 1
    elif np.isclose(action.action_num, 0.23):
        ac = 2
    else:
        raise ValueError(f'Invalid action : {action.action_num}')
    return ac

def mountain_car_rescale(observation:np.ndarray)->np.ndarray:
    x_min, x_max = -1.2, 0.5
    v_min, v_max = -0.07, 0.07
    x, v = observation[0], observation[1]

    x_scaled = (x - x_min)/(x_max - x_min)
    v_scaled = (v - v_min)/(v_max - v_min)

    return np.array([x_scaled, v_scaled])

def main():

    run_config = {
        'mdp' : 2,
        'algo' : 2,
        'train' : True, # Otherwise load (and no save)
        'save' : True,
        'demo' : False   # Otherwise graph
    }

    # Basic environment details
    if run_config['mdp']==1:
        env_name = 'CartPole-v1'
        action_space = [Action(0.1), Action(0.17)]
        state_shape = 4
        gamma = 1.0
        obs_modifier_func = None
        ac_modifier_func = action_mapping
        save_dir = 'cartpole'
    elif run_config['mdp']==2:
        env_name = 'MountainCar-v0'
        action_space = [Action(0.1), Action(0.17), Action(0.23)]#[Action(i) for i in range(3)]
        state_shape = 2 #6 if obs_modifier_func is None else 4
        gamma = 1.0
        obs_modifier_func = mountain_car_rescale
        ac_modifier_func = action_mapping
        save_dir = 'mountain_car'
    else:
        raise ValueError(f'Invalid mdp number: {run_config["mdp"]}')

    env = gym.make(env_name)

    # Model parameters
    SAR = Fourier_SAR
    # SAR = FourierInd_SAR
    SAR_order = 2
    SR_order = 2
    SAR_args = {'order':SAR_order}
    SR = Fourier_SR
    # SR = FourierInd_SR
    SR_args = {'order':SR_order}

    # Initializing model
    SAR.reset()
    SR.reset()
    SAR_weights = SAR.init_weights(state_shape, method='normal', order=SAR_order)
    SR_weights = SR.init_weights(state_shape, order=SR_order)

    # Training parameters
    max_ep = 5_000
    alpha = 0.005
    alpha_w = 0.001
    alpha_theta = 0.001
    # epsilon_schedule = lambda x: 0.95/((10*x)/max_ep + 1)+ 0.05
    epsilon_schedule = lambda x: 0.95/((100*x)/max_ep + 1)+ 0.05
    # epsilon_schedule = lambda x: 0.05
    # temperature_schedule = lambda x: 2.0/((100*x)/max_ep + 1)+ 0.2
    temperature_schedule = lambda x: 1.0/((100*x)/max_ep + 1)+ 0.2
    temperature_schedule = lambda x: 1.0
    n = 1

    #Experiments
    n_arr = [n//2, n, 2*n]
    n_arr = [n]
    alpha_arr = [alpha/10, alpha, alpha*10]
    alpha_arr = [ 0.01, 0.001, 0.005, 0.0005]
    alpha_w_arr = [alpha_w/10, alpha_w, alpha_w*10]
    alpha_theta_arr = [alpha_theta/10, alpha_theta, alpha_theta*10]
    num_exp = 10

    #Single Runs
    # n_arr = [n]
    # alpha_arr = [alpha]
    # alpha_w_arr = [alpha_w]
    # alpha_theta_arr = [alpha_theta]

    # Analysis tools
    save_loc = f'data/saved_models/{save_dir}/demo_model_{run_config["mdp"]}_{run_config["algo"]}.npz'

    if run_config['train']:
        if run_config['algo'] == 1:
            # Semi grad sarsa
            print(f'Algo 1')
            actions_list = []
            steps_list = []
            for exp in range(num_exp):
                q_func = QFunc(SAR_weights, action_space)
                q_func, data = semi_grad_sarsa(
                                    env = env,
                                    q_func = q_func,
                                    SAR = SAR,
                                    SAR_args=SAR_args,
                                    gamma = gamma,
                                    alpha = alpha,
                                    epsilon_schedule = epsilon_schedule,
                                    n = n,
                                    state_shape = state_shape,
                                    obs_modifier_func = obs_modifier_func,
                                    ac_modifier_func = ac_modifier_func,
                                    max_ep = max_ep)
                steps = data['num_steps']
                actions = np.cumsum(steps)
                steps_list.append(steps)
                actions_list.append(actions)
            plot_steps_v3(actions_list, steps_list)

            # all_results_n = {}
            # for n_exp in n_arr:
            #     q_func = QFunc(SAR_weights, action_space)
            #     q_func, data = semi_grad_sarsa(
            #                         env = env,
            #                         q_func = q_func,
            #                         SAR = SAR,
            #                         SAR_args=SAR_args,
            #                         gamma = gamma,
            #                         alpha = alpha,
            #                         epsilon_schedule = epsilon_schedule,
            #                         n = n_exp,
            #                         state_shape = state_shape,
            #                         obs_modifier_func = obs_modifier_func,
            #                         ac_modifier_func = ac_modifier_func,
            #                         max_ep = max_ep)
            #     all_results_n[n_exp] = data['num_steps']
            # print("Plots for varying n")
            # plot_steps_v2(all_results_n, variable = 'n',  N=100, title='steps')

            # all_results_alpha = {}
            # for alpha_exp in alpha_arr:
            #     q_func = QFunc(SAR_weights, action_space)
            #     q_func, data = semi_grad_sarsa(
            #                         env = env,
            #                         q_func = q_func,
            #                         SAR = SAR,
            #                         SAR_args=SAR_args,
            #                         gamma = gamma,
            #                         alpha = alpha_exp,
            #                         epsilon_schedule = epsilon_schedule,
            #                         n = n,
            #                         state_shape = state_shape,
            #                         obs_modifier_func = obs_modifier_func,
            #                         ac_modifier_func = ac_modifier_func,
            #                         max_ep = max_ep)
            #     all_results_alpha[alpha_exp] = data['num_steps']
            # print("Plots for varying alpha")
            # plot_steps_v2(all_results_alpha, variable = 'alpha',  N=100, title='steps')

            # Broken
            if run_config['save']:
                save_func_data(q_func.weights, data=data, save_loc=save_loc)
        
        elif run_config['algo'] == 2:
            # reinforce
            print('Algo 2')

            actions_list = []
            steps_list = []
            for exp in range(num_exp):
                v_func = ValueFunc(SR_weights)
                policy_func = PolicyFunc(SAR_weights, action_space)
                policy_func, v_func, data = reinforce_with_baseline(
                                    env = env,
                                    v_func = v_func,
                                    policy_func = policy_func,
                                    SAR = SAR,
                                    SAR_args = SAR_args,
                                    SR = SR,
                                    SR_args = SR_args,
                                    temperature_schedule = temperature_schedule,
                                    gamma = gamma,
                                    alpha_w = alpha_w,
                                    alpha_theta = alpha_theta,
                                    obs_modifier_func = obs_modifier_func,
                                    ac_modifier_func = ac_modifier_func,
                                    max_ep = max_ep)
                steps = data['num_steps']
                actions = np.cumsum(steps)
                steps_list.append(steps)
                actions_list.append(actions)
            plot_steps_v3(actions_list, steps_list)

            # all_results_alpha_w = {}
            # for alpha_w_exp in alpha_w_arr:
            #     v_func = ValueFunc(SR_weights)
            #     policy_func = PolicyFunc(SAR_weights, action_space)
            #     policy_func, v_func, data = reinforce_with_baseline(
            #                         env = env,
            #                         v_func = v_func,
            #                         policy_func = policy_func,
            #                         SAR = SAR,
            #                         SAR_args = SAR_args,
            #                         SR = SR,
            #                         SR_args = SR_args,
            #                         temperature_schedule = temperature_schedule,
            #                         gamma = gamma,
            #                         alpha_w = alpha_w_exp,
            #                         alpha_theta = alpha_theta,
            #                         obs_modifier_func = obs_modifier_func,
            #                         ac_modifier_func = ac_modifier_func,
            #                         max_ep = max_ep)
            #     all_results_alpha_w[alpha_w_exp] = data['num_steps']
            # print("Plots for varying alpha_w")
            # plot_steps_v2(all_results_alpha_w, variable = 'alpha_w',  N=100, title='steps')

            # all_results_alpha_theta = {}
            # for alpha_theta_exp in alpha_theta_arr:
            #     v_func = ValueFunc(SR_weights)
            #     policy_func = PolicyFunc(SAR_weights, action_space)
            #     policy_func, v_func, data = reinforce_with_baseline(
            #                         env = env,
            #                         v_func = v_func,
            #                         policy_func = policy_func,
            #                         SAR = SAR,
            #                         SAR_args = SAR_args,
            #                         SR = SR,
            #                         SR_args = SR_args,
            #                         temperature_schedule = temperature_schedule,
            #                         gamma = gamma,
            #                         alpha_w = alpha_w,
            #                         alpha_theta = alpha_theta_exp,
            #                         obs_modifier_func = obs_modifier_func,
            #                         ac_modifier_func = ac_modifier_func,
            #                         max_ep = max_ep)
            #     all_results_alpha_theta[alpha_theta_exp] = data['num_steps']
            # print("Plots for varying alpha_theta")
            # plot_steps_v2(all_results_alpha_theta, variable = 'alpha_theta',  N=100, title='steps')
            
            # Broken
            if run_config['save']:
                save_func_data(policy_func.weights, v_func.weights, data=data, save_loc=save_loc)
        
        elif run_config['algo'] == 3:
            # actor critic
            print('Algo 3')

            actions_list = []
            steps_list = []
            for exp in range(num_exp):
                v_func = ValueFunc(SR_weights)
                policy_func = PolicyFunc(SAR_weights, action_space)
                policy_func, v_func, data = one_step_actor_critic(
                                    env = env,
                                    v_func = v_func,
                                    policy_func = policy_func,
                                    SAR = SAR,
                                    SAR_args = SAR_args,
                                    SR = SR,
                                    SR_args = SR_args,
                                    temperature_schedule = temperature_schedule,
                                    gamma = gamma,
                                    alpha_w = alpha_w,
                                    alpha_theta = alpha_theta,
                                    obs_modifier_func = obs_modifier_func,
                                    ac_modifier_func = ac_modifier_func,
                                    max_ep = max_ep)
                steps = data['num_steps']
                actions = np.cumsum(steps)
                steps_list.append(steps)
                actions_list.append(actions)
            plot_steps_v3(actions_list, steps_list)

            # all_results_alpha_w = {}
            # for alpha_w_exp in alpha_w_arr:
            #     v_func = ValueFunc(SR_weights)
            #     policy_func = PolicyFunc(SAR_weights, action_space)
            #     policy_func, v_func, data = one_step_actor_critic(
            #                         env = env,
            #                         v_func = v_func,
            #                         policy_func = policy_func,
            #                         SAR = SAR,
            #                         SAR_args = SAR_args,
            #                         SR = SR,
            #                         SR_args = SR_args,
            #                         temperature_schedule = temperature_schedule,
            #                         gamma = gamma,
            #                         alpha_w = alpha_w_exp,
            #                         alpha_theta = alpha_theta,
            #                         obs_modifier_func = obs_modifier_func,
            #                         ac_modifier_func = ac_modifier_func,
            #                         max_ep = max_ep)
            #     all_results_alpha_w[alpha_w_exp] = data['num_steps']
            # print("Plots for varying alpha_w")
            # plot_steps_v2(all_results_alpha_w, variable = 'alpha_w',  N=100, title='steps')

            # all_results_alpha_theta = {}
            # for alpha_theta_exp in alpha_theta_arr:
            #     v_func = ValueFunc(SR_weights)
            #     policy_func = PolicyFunc(SAR_weights, action_space)
            #     policy_func, v_func, data = one_step_actor_critic(
            #                         env = env,
            #                         v_func = v_func,
            #                         policy_func = policy_func,
            #                         SAR = SAR,
            #                         SAR_args = SAR_args,
            #                         SR = SR,
            #                         SR_args = SR_args,
            #                         temperature_schedule = temperature_schedule,
            #                         gamma = gamma,
            #                         alpha_w = alpha_w,
            #                         alpha_theta = alpha_theta_exp,
            #                         obs_modifier_func = obs_modifier_func,
            #                         ac_modifier_func = ac_modifier_func,
            #                         max_ep = max_ep)
            #     all_results_alpha_theta[alpha_theta_exp] = data['num_steps']
            # print("Plots for varying alpha_theta")
            # plot_steps_v2(all_results_alpha_theta, variable = 'alpha_theta',  N=100, title='steps')
            
            # broken
            if run_config['save']:
                save_func_data(policy_func.weights, v_func.weights, data=data, save_loc=save_loc)
    
    else:
        if run_config['algo']==1:
            arrays, data = load_func(1, save_loc=save_loc)
            q_func = QFunc(arrays[0], action_space)
        elif run_config['algo'] in [2, 3]:
            arrays, data = load_func(2, save_loc=save_loc)
            policy_func = PolicyFunc(arrays[0], action_space)
            v_func = ValueFunc(arrays[1])

    if run_config['demo']:
        if run_config['algo']==1:
            demo(gym.make(env_name, render_mode='human'), q_func, SAR, SAR_args)
        elif run_config['algo'] in [2, 3]:
            demo_new(gym.make(env_name, render_mode='human'), policy_func, SAR, SAR_args)
    else:
        try:
            plot_steps(data['num_steps'], N=100, title='steps')
        except KeyError:
            logging.warning('Can\'t find key "num_steps" in data')
        try:
            plot_steps(data['rewards'], N=100, title='rewards')
        except KeyError:
            logging.warning('Can\'t find key "rewards"')

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    main()