import numpy as np
import gym
from typing import Callable, Type
from tqdm import trange
import logging

from utils.helpers import StateRepresentation, StateActionRepresentation, Action
from utils.containers import QFunc, PolicyFunc, PolicyFunc2, ValueFunc

def one_step_actor_critic(
        env:gym.Env,
        v_func:ValueFunc,
        policy_func:PolicyFunc,
        SAR:Type[StateActionRepresentation],
        SAR_args:dict,
        SR:Type[StateRepresentation],
        SR_args:dict,
        temperature_schedule:Callable[[int], float],
        gamma:float,
        alpha_w:float,
        alpha_theta:float,
        obs_modifier_func:Callable[[np.ndarray], np.ndarray]=None,
        ac_modifier_func:Callable[[Action], int]=None,
        max_ep:int=10_000
    )-> tuple[PolicyFunc, ValueFunc, dict]:
    
    # Supplemental output
    data = dict()

    # Record number of steps and total reward per episode
    num_steps = np.zeros((max_ep,))
    rewards = np.zeros((max_ep,))

    pbar = trange(max_ep)
    for ep in pbar:
        temperature = temperature_schedule(ep)

        # Reward tracker
        tot_reward = 0

        # Init episode
        observation, info = env.reset()
        observation_original = observation

        # If state vector needs to be modified
        if obs_modifier_func is not None:
            observation = obs_modifier_func(observation)
        
        I = 1.

        # Run the episode till termination
        truncated = terminated = False
        step = 0
        while not (truncated or terminated):
            action = policy_func.sample(SAR, SAR_args, StateRepresentation(observation), temperature)
            
            # Take the action
            if ac_modifier_func is None:
                ac = int(action.action_num)
            else:
                ac = ac_modifier_func(action)

            observation_, reward, terminated, truncated, info = env.step(ac)
            observation_original_ = observation_

            # If state vector needs to be modified
            if obs_modifier_func is not None:
                observation_ = obs_modifier_func(observation_)

            # # Reward modification
            # reward = 100*((np.sin(3 * observation_original_[0]) * 0.0025 + 0.5 * observation_original_[1] * observation_original_[1]) - (np.sin(3 * observation_original[0]) * 0.0025 + 0.5 * observation_original[1] * observation_original[1]))
            # if observation_original_[0] >= 0.5:
            #     reward += 1
            
            tot_reward += reward

            # Only terminated?
            v_ = 0. if (terminated or truncated) else v_func(SR(observation, **SR_args))
            s = SR(observation, **SR_args)
            delta = reward + gamma * v_ - v_func(s)
            v_func.increment_weights(alpha_w * delta * v_func.grad(s))
            
            sv_t = StateRepresentation(observation)
            term2 = policy_func.grad_term(SAR, SAR_args, sv_t, temperature)
            grad_ln = (1/temperature) * (SAR(sv_t, action, **SAR_args).get_arr() - term2)
            inc = alpha_theta * I * delta * grad_ln
            policy_func.increment_weights(inc)
            I = gamma * I
            observation = observation_
            step += 1
            # print(inc)


        rewards[ep] = tot_reward
        num_steps[ep] = step

        if ep % 20 == 0:
            pbar.set_description(f'Total steps: {step:4}, Total reward: {tot_reward:8.2f}, wt1: {policy_func.weights[-5]:.4f}, vwt1: {v_func.weights[1]:.4f}')

    pbar.close()
    env.close()

    data['num_steps'] = num_steps
    data['rewards'] = rewards
    
    return policy_func, v_func, data


def one_step_actor_critic_new(
        env:gym.Env,
        v_func:ValueFunc,
        policy_func:PolicyFunc2,
        SR:Type[StateRepresentation],
        SR_args:dict,
        temperature_schedule:Callable[[int], float],
        gamma:float,
        alpha_w:float,
        alpha_theta:float,
        obs_modifier_func:Callable[[np.ndarray], np.ndarray]=None,
        max_ep:int=10_000
    )-> tuple[PolicyFunc, ValueFunc, dict]:
    
    # Supplemental output
    data = dict()

    # Record number of steps and total reward per episode
    num_steps = np.zeros((max_ep,))
    rewards = np.zeros((max_ep,))

    pbar = trange(max_ep)
    for ep in pbar:
        temperature = temperature_schedule(ep)

        # Reward tracker
        tot_reward = 0

        # Init episode
        observation, info = env.reset()
        observation_original = observation

        # If state vector needs to be modified
        if obs_modifier_func is not None:
            observation = obs_modifier_func(observation)
        
        I = 1.

        # Run the episode till termination
        truncated = terminated = False
        step = 0
        while not (truncated or terminated):
            phi = SR(observation, **SR_args)
            action = policy_func.sample(phi)
            
            # Take the action
            if np.isclose(action.action_num, 0.1):
                ac = 0
            elif np.isclose(action.action_num, 0.23):
                ac = 2
            else:
                raise ValueError(f'Invalid action : {action.action_num}')
            observation_, reward, terminated, truncated, info = env.step(int(action.action_num))
            observation_original_ = observation_

            # If state vector needs to be modified
            if obs_modifier_func is not None:
                observation_ = obs_modifier_func(observation_)

            # # Reward modification
            # reward = 100*((np.sin(3 * observation_original_[0]) * 0.0025 + 0.5 * observation_original_[1] * observation_original_[1]) - (np.sin(3 * observation_original[0]) * 0.0025 + 0.5 * observation_original[1] * observation_original[1]))
            # if observation_original_[0] >= 0.5:
            #     reward += 1
            
            tot_reward += reward

            # # Only terminated?
            # v_ = 0. if (terminated or truncated) else v_func(SR(observation, **SR_args))
            # s = SR(observation, **SR_args)
            # delta = reward + gamma * v_ - v_func(s)
            # v_func.increment_weights(alpha_w * delta * v_func.grad(s))
            
            # sv_t = StateRepresentation(observation)
            # term2 = policy_func.grad_term(SAR, SAR_args, sv_t, temperature)
            # grad_ln = (1/temperature) * (SAR(sv_t, action, **SAR_args).get_arr() - term2)
            # inc = alpha_theta * I * delta * grad_ln
            # policy_func.increment_weights(inc)
            # I = gamma * I
            # observation = observation_
            # step += 1
            # print(inc)


        rewards[ep] = tot_reward
        num_steps[ep] = step

        if ep % 20 == 0:
            pbar.set_description(f'Total steps: {step:4}, Total reward: {tot_reward:6.2f}, wt1: {policy_func.weights[-5]:.4f}, vwt1: {v_func.weights[1]}')

    pbar.close()
    env.close()

    data['num_steps'] = num_steps
    data['rewards'] = rewards
    
    return policy_func, v_func, data