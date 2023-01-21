import numpy as np
from typing import Type

class Action:
    def __init__(self, action_num:float):
        self.action_num = action_num

    def __repr__(self):
        return str(self.action_num)

class Representation:
    def __init__(self, arr:np.ndarray):
        self.arr = arr

    def get_arr(self)->np.ndarray:
        return self.arr

    def set_arr(self, arr:np.ndarray):
        self.arr = arr

    def __repr__(self):
        return repr(self.arr)

class StateRepresentation(Representation):
    def __init__(self, state_arr:np.ndarray):
        super(StateRepresentation, self).__init__(state_arr)

    @classmethod
    def get_weight_size(cls, state_size:int):
        return state_size

    @classmethod
    def init_weights(cls, state_size:int, method:str='zeros', **params):
        if method=='zeros':
            return np.zeros((cls.get_weight_size(state_size, **params),))

        # If method is in none of the options
        raise ValueError(f'Invalid value for `method`: {method}')

class StateActionRepresentation(Representation):
    def __init__(self, state_rep:StateRepresentation, action:Action):
        self.state_rep = state_rep
        self.action = action
        super(StateActionRepresentation, self).__init__(self.rep())

    def rep(self):
        return np.array([*self.state_rep.arr, self.action.action_num])

    @classmethod
    def get_weight_size(cls, state_size:int):
        return state_size+1

    @classmethod
    def init_weights(cls, state_size:int, method:str='zeros', **params):
        if method=='zeros':
            return np.zeros((cls.get_weight_size(state_size, **params),))

        if method=='ones':
            return np.ones((cls.get_weight_size(state_size, **params),))

        if method=='normal':
            return np.random.normal(0., 0.1, size=(cls.get_weight_size(state_size, **params),))

        # If method is in none of the options
        raise ValueError(f'Invalid value for `method`: {method}')

    @classmethod
    def reset(cls):
        # Make it mandatory to run before instantiating objects
        pass

class Parametrized:
    def __init__(self, weights:np.ndarray, action_space:list[Action]):
        self.weights = weights
        self.action_space = action_space
        self.rng = np.random.default_rng()

    def __call__(self, inp:Representation)->float:
        return np.dot(self.weights, inp.get_arr())

    def grad(self, inp:Representation)->np.ndarray:
        return inp.get_arr()

    def increment_weights(self, inc:np.ndarray)->None:
        self.weights = self.weights + inc


    