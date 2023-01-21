import numpy as np
from typing import Type
from scipy.special import softmax, logsumexp
import logging

from utils.helpers import StateRepresentation, StateActionRepresentation, Action, Parametrized

class QFunc(Parametrized):
    def __init__(self, weights:np.ndarray, action_space:list[Action]):
        super(QFunc, self).__init__(weights, action_space)

    def greedy(self,
            SAR:Type[StateActionRepresentation],
            SAR_args:dict,
            state:StateRepresentation
        )->Action:
        best_q_est = -np.inf
        best_action = None
        for action in self.action_space:
            q_est = self(SAR(state, action, **SAR_args))
            if q_est > best_q_est:
                best_q_est = q_est
                best_action = action

        if best_action is None:
            logging.warning('Best action is None')

            for action in self.action_space:
                logging.info(f'action value: {self(SAR(state, action, **SAR_args))}')

        return best_action

    def e_greedy(self,
            SAR:Type[StateActionRepresentation],
            SAR_args:dict,
            state:StateRepresentation,
            epsilon:float
        )->Action:
        # When we want to explore
        p = self.rng.uniform(low=0., high=1.)
        if p < epsilon:
            return np.random.choice(self.action_space)

        # When we do not want to explore (exploit)
        return self.greedy(SAR, SAR_args, state)

    

class Fourier_SAR(StateActionRepresentation):
    def __init__(self, state_rep:StateRepresentation, action:Action, order:int):
        self.order = order
        try:
            x = __class__.weight_size
        except AttributeError:
            __class__.reset()

        super(Fourier_SAR, self).__init__(state_rep, action)

    def rep(self):
        raw_feats = np.array([*self.state_rep.arr, self.action.action_num])
        arr = __class__.get_base_arr(len(self.state_rep.arr), self.order)
        arr = np.cos(np.pi * np.multiply(arr, raw_feats).sum(axis=1))
        return arr

    @classmethod
    def get_weight_size(cls, state_size:int, order:int):
        if cls.weight_size is None:
            cls.weight_size = (order+1)**(state_size+1)

        return cls.weight_size

    @classmethod
    def get_base_arr(cls, state_size:int, order:int):
        if cls.base_arr is None:
            Z = cls.get_weight_size(state_size, order)
            arr = np.array(range(Z))
            arr = np.broadcast_to(arr, (state_size+1, Z)).T
            powers = (order+1)**np.array(range(state_size+1))
            cls.base_arr = (arr // powers) % (order+1)

        return cls.base_arr

    @classmethod
    def get_scale_factor(cls, state_size:int, order:int):
        if cls.scale_factor is None:
            Z = cls.get_weight_size(state_size, order)
            base_arr = cls.get_base_arr(state_size, order)
            eta = np.linalg.norm(base_arr, ord=2, axis=1)
            scale = np.ones((Z,))
            scale[1:] = 1/eta[1:]
            cls.scale_factor = scale
        
        return cls.scale_factor

    @classmethod
    def reset(cls):
        cls.weight_size = None
        cls.base_arr = None
        cls.scale_factor = None

class FourierInd_SAR(StateActionRepresentation):
    def __init__(self, state_rep:StateRepresentation, action:Action, order:int):
        self.order = order
        super(FourierInd_SAR, self).__init__(state_rep, action)

    def rep(self):
        N = len(self.state_rep.arr)
        Z = __class__.get_weight_size(N, self.order)
        arr = np.zeros((Z,))
        M = self.order + 1

        arr[0] = 1.
        for i in range(1, M): # Order
            # All state features
            for j in range(N): # Feature
                arr[i + j*(M-1)] = np.cos(self.state_rep.arr[j] * i * np.pi)

            # Action
            arr[i + N*(M-1)] = np.cos(self.action.action_num * i * np.pi)

        return arr

    @classmethod
    def get_weight_size(cls, state_size:int, order:int):
        return order*(state_size+1) + 1

class Poly_SAR(StateActionRepresentation):
    def __init__(self, state_rep:StateRepresentation, action:Action, order:int):
        self.order = order
        super(Poly_SAR, self).__init__(state_rep, action)

    def rep(self):
        N = len(self.state_rep.arr)
        Z = __class__.get_weight_size(N, self.order)
        arr = np.zeros((Z,))
        M = self.order + 1

        arr[0] = 1.
        for i in range(1, M): # Order
            # All state features
            for j in range(N): # Feature
                arr[i + j*(M-1)] = self.state_rep.arr[j]**i

            # Action
            arr[i + N*(M-1)] = self.action.action_num**i

        return arr

    @classmethod
    def get_weight_size(cls, state_size:int, order:int):
        return order*(state_size+1) + 1

class PolicyFunc(Parametrized):
    def __init__(self, weights:np.ndarray, action_space:list[Action]):
        super(PolicyFunc, self).__init__(weights, action_space)

    def sample_(self,
            SAR:Type[StateActionRepresentation],
            SAR_args:dict,
            state:StateRepresentation,
            temperature:float
        )->Action:
        all_prefs = np.array([self(SAR(state, action, **SAR_args)) for action in self.action_space])
        probs = softmax(all_prefs/temperature)
        return self.rng.choice(self.action_space, p=probs)

    def sample(self,
            SAR:Type[StateActionRepresentation],
            SAR_args:dict,
            state:StateRepresentation,
            temperature:float
        )->Action:
        D = self.D(self.Phi(SAR, SAR_args, state))
        probs = softmax(D/temperature)
        return self.rng.choice(self.action_space, p=probs)

    def Phi(self,
            SAR:Type[StateActionRepresentation],
            SAR_args:dict,
            state:StateRepresentation
        ):
        sars = [SAR(state, action, **SAR_args).get_arr() for action in self.action_space]
        return np.array(sars)

    def D(self, Phi:np.ndarray):
        return np.matmul(Phi, self.weights)

    def grad_term(self,
            SAR:Type[StateActionRepresentation],
            SAR_args:dict,
            state:StateRepresentation,
            temperature:float
        )->np.ndarray:
        Phi = self.Phi(SAR, SAR_args, state)
        D = self.D(Phi)
        _1_tD = (1/temperature)*D
        n_B_expon = np.broadcast_to(_1_tD, Phi.T.shape).T
        n_lse, n_signs = logsumexp(n_B_expon, axis=0, b=Phi, return_sign=True)
        d_lse = logsumexp(_1_tD)
        # print('n')
        # print(n_lse)
        # print('d')
        # print(d_lse)
        # print('o')
        ret = np.multiply(np.exp(n_lse - d_lse), n_signs)
        return ret

    # def increment_weights(self, inc:np.ndarray, SAR:Type[StateActionRepresentation], state_size, order)->None:
    #     scale_factor = SAR.get_scale_factor(state_size, order)
    #     self.weights = self.weights + inc

class ValueFunc(Parametrized):
    def __init__(self, weights:np.ndarray):
        self.weights = weights

class FourierInd_SR(StateRepresentation):
    def __init__(self, state_arr:np.ndarray, order:int):        
        self.order = order
        super(FourierInd_SR, self).__init__(self.rep(state_arr))

    def rep(self, state_arr:np.ndarray):
        N = len(state_arr)
        Z = __class__.get_weight_size(N, self.order)
        arr = np.zeros((Z,))
        M = self.order + 1

        arr[0] = 1.
        for i in range(1, M): # Order
            # All state features
            for j in range(N): # Feature
                arr[i + j*(M-1)] = np.cos(state_arr[j] * i * np.pi)

        return arr

    @classmethod
    def get_weight_size(cls, state_size:int, order:int):
        return order*(state_size) + 1

    @classmethod
    def reset(*args, **kwargs):
        pass

class Fourier_SR(StateRepresentation):
    def __init__(self, state_arr:np.ndarray, order:int):
        try:
            x = __class__.weight_size
        except AttributeError:
            __class__.reset()
        
        self.order = order
        super(Fourier_SR, self).__init__(self.rep(state_arr))

    # TODO: Use these from a Fourier super class
    def rep(self, state_arr:np.ndarray):
        arr = __class__.get_base_arr(len(state_arr), self.order)
        arr = np.cos(np.pi * np.multiply(arr, state_arr).sum(axis=1))
        return arr

    @classmethod
    def get_weight_size(cls, state_size:int, order:int):
        if cls.weight_size is None:
            cls.weight_size = (order+1)**state_size

        return cls.weight_size

    @classmethod
    def get_base_arr(cls, state_size:int, order:int):
        if cls.base_arr is None:
            Z = cls.get_weight_size(state_size, order)
            arr = np.array(range(Z))
            arr = np.broadcast_to(arr, (state_size, Z)).T
            powers = (order+1)**np.array(range(state_size))
            cls.base_arr = (arr // powers) % (order+1)

        return cls.base_arr

    @classmethod
    def reset(cls):
        cls.weight_size = None
        cls.base_arr = None

class PolicyFunc2(Parametrized):
    def __init__(self, weights:np.ndarray, action_space:list[Action]):
        self.A = len(action_space)
        super(PolicyFunc2, self).__init__(weights, action_space)

    def sample(self,
            state:StateRepresentation
        )->Action:
        D = np.matmul(self.weights, state.get_arr())
        ps = softmax(D)
        return self.rng.choice(self.action_space, p=ps)

    def __call__(self, state:StateRepresentation, action:Action)->float:
        for i in range(self.A):
            if np.isclose(self.action_space[i].action_num, action.action_num):
                return softmax(np.matmul(self.weights, state.get_arr()))[i]

    def grad_ln(self, state:StateRepresentation, action:Action):
        grad = np.zeros_like(self.weights)
        ps = softmax(np.matmul(self.weights, state.get_arr()))
        a_cnt = 0
        for i in range(self.A):
            # if a_i == a_j
            if np.isclose(self.action_space[i].action_num, action.action_num):
                grad[i, :] = (1 - ps[i]) * state.get_arr()
                a_cnt += 1
            # if a_i != a_j
            else:
                grad[i, :] = - ps[i] * state.get_arr()

        assert a_cnt == 1, f'Action present {a_cnt} times in action_space'
        return grad
    