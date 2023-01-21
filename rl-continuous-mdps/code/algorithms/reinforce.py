import numpy as np
import gym
from typing import Callable, Type
from tqdm import trange
import logging

from utils.helpers import StateRepresentation, StateActionRepresentation, Action
from utils.containers import QFunc, PolicyFunc, ValueFunc

def reinforce_with_baseline(
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

        # Containers
        S_arr = []
        A_arr = []
        R_arr = []

        # Reward tracker
        tot_reward = 0

        # Generate an episode
        observation, info = env.reset()

        # If state vector needs to be modified
        if obs_modifier_func is not None:
            observation = obs_modifier_func(observation)

        truncated = terminated = False
        while not (truncated or terminated):
            action = policy_func.sample(SAR, SAR_args, StateRepresentation(observation), temperature)
            S_arr.append(observation)
            A_arr.append(action)

            # Take the action
            if ac_modifier_func is None:
                ac = int(action.action_num)
            else:
                ac = ac_modifier_func(action)
            observation, reward, terminated, truncated, info = env.step(ac)
            
            # If state vector needs to be modified
            if obs_modifier_func is not None:
                observation = obs_modifier_func(observation)

            R_arr.append(reward)
            tot_reward += reward

        T = len(S_arr)
        assert T == len(A_arr) == len(R_arr), 'Something is not right.'

        # Record supplemental data
        num_steps[ep] = T
        rewards[ep] = tot_reward

        # if truncated:
        #     continue

        # Make updates based on recorded episode
        for t in range(T):
            sr_t = SR(S_arr[t], **SR_args)
            sv_t = StateRepresentation(S_arr[t])
            sar_t = SAR(sv_t, A_arr[t], **SAR_args)

            # Set G
            G = 0.
            for k in range(t+1, T+1):
                G += (gamma**(k-t-1)) * R_arr[k-1]  # R_k is at index k-1

            # Set delta
            delta = G - v_func(sr_t)

            # update v_func wts
            v_func.increment_weights(alpha_w * delta * v_func.grad(sr_t))

            # # update policy_func wts
            # inc = alpha_theta * (gamma**t) * delta
            # inc = inc * policy_func.grad(sar_t) / ( policy_func(sar_t) + 1e-15 )
            # policy_func.increment_weights(inc)

            # update policy_func wts
            term2 = policy_func.grad_term(SAR, SAR_args, sv_t, temperature)
            grad_ln = (1/temperature) * (sar_t.get_arr() - term2)
            if not np.isfinite(grad_ln).all():
                logging.warning('Invalid policy gradient')
            policy_func.increment_weights(alpha_theta*(gamma**t)*delta*grad_ln)

        if ep % 20 == 0:
            pbar.set_description(f'Num steps: {T:4}, Total reward: {tot_reward:8.2f}')

    pbar.close()
    env.close()

    data['num_steps'] = num_steps
    data['rewards'] = rewards
    
    return policy_func, v_func, data