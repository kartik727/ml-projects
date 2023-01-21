import numpy as np
import gym
from typing import Callable, Type
from tqdm import trange

from utils.helpers import StateRepresentation, StateActionRepresentation, Action
from utils.containers import QFunc

def semi_grad_sarsa(
        env:gym.Env,
        q_func:QFunc,
        SAR:Type[StateActionRepresentation],
        SAR_args:dict,
        gamma:float,
        alpha:float,
        epsilon_schedule:Callable[[int], float],
        n:int,
        state_shape:int,
        obs_modifier_func:Callable[[np.ndarray], np.ndarray]=None,
        ac_modifier_func:Callable[[Action], int]=None,
        max_ep:int=10_000
    )-> tuple[QFunc, dict]:
    S_arr = np.zeros((n+1, state_shape))
    A_arr = np.zeros((n+1,))
    R_arr = np.zeros((n+1,))

    # Supplemental output
    data = dict()

    # Record number of steps per episode
    num_steps = np.zeros((max_ep,))

    pbar = trange(max_ep)
    for ep in pbar:
        epsilon = epsilon_schedule(ep)
        observation, info = env.reset()

        # If state vector needs to be modified
        if obs_modifier_func is not None:
            observation = obs_modifier_func(observation)
        
        S_arr[0, :] = observation
        A_arr[0] = q_func.e_greedy(SAR, SAR_args, StateRepresentation(observation), epsilon).action_num

        T = np.inf
        t = 0
        while True:
            if t < T:
                if ac_modifier_func is None:
                    ac = int(Action(A_arr[t % (n+1)]).action_num)
                else:
                    ac = ac_modifier_func(Action(A_arr[t % (n+1)]))
                observation, reward, terminated, truncated, info = env.step(ac)
                # If state vector needs to be modified
                if obs_modifier_func is not None:
                    observation = obs_modifier_func(observation)

                R_arr[(t+1) % (n+1)] = reward
                S_arr[(t+1) % (n+1), :] = observation
                if terminated or truncated:
                    T = t + 1
                else:
                    A_arr[(t+1) % (n+1)] = q_func.e_greedy(SAR, SAR_args, StateRepresentation(observation), epsilon).action_num

            tau = t - n + 1
            if tau>=0:
                G = 0.
                for i in range(tau+1, min(tau+n, T)+1):
                    G += (gamma**(i-tau-1)) * R_arr[i % (n+1)]
                    
                if tau+n<T:
                    G += (gamma**n) * q_func(SAR(
                        StateRepresentation(S_arr[(tau+n) % (n+1), :]),
                        Action(A_arr[(tau+n) % (n+1)]),
                        **SAR_args
                    ))

                # Update weights
                sar = SAR(
                    StateRepresentation(S_arr[tau % (n+1), :]),
                    Action(A_arr[tau % (n+1)]),
                    **SAR_args
                )
                inc = alpha * (G - q_func(sar)) * q_func.grad(sar)
                q_func.increment_weights(inc)

            # Checks for end of iteration
            if tau == T - 1:
                break
            t += 1

        num_steps[ep] = t
        if ep % 10 == 0:
            pbar.set_description(f'Num steps: {t:4}')

    pbar.close()
    env.close()

    
    data['num_steps'] = num_steps
    
    return q_func, data