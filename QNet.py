import numpy as np
import scipy.optimize
import sparse_ffnn as nn
from random import uniform, randint

class QNet:

    def __init__(self,input_size,num_actions,hidden_size):
        self.sparsity_param = 0.1
        self.lambda_ = 3e-3
        self.beta = 3
        self.gamma = 1
        self.epsilon = 0.5
        self.input_size = input_size
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.theta = nn.initialize(self.input_size, self.hidden_size, self.num_actions)
        self.options_ = {'maxiter': 400, 'disp': True}
        self.previous_state = np.zeros(self.input_size).reshape(self.input_size,1)
        self.previous_total_reward = 0

    def train(self, state, action, total_reward, terminal):
        if not terminal:
            pred = nn.predict(self.theta, self.input_size, self.hidden_size, self.num_actions, state)
            reward = total_reward - self.previous_total_reward
            J = lambda x: nn.cost(
                x, self.input_size, self.hidden_size, self.num_actions, self.lambda_, self.sparsity_param, self.beta,
                self.previous_state, reward + self.gamma * pred.reshape(self.num_actions,1)
            )
        else:
            J = lambda x: nn.cost(
                x, self.input_size, self.hidden_size, self.num_actions, self.lambda_, self.sparsity_param, self.beta,
                self.previous_state, reward
            ) 
        result = scipy.optimize.minimize(
            J, self.theta, method='L-BFGS-B', jac=True, options=self.options_
        )
        self.theta = result.x
        self.previous_state = state

    def get_action(self, state):
        if uniform(0,1) < self.epsilon:
            action = randint(0, self.num_actions-1)
        else:
            action = np.argmax(nn.predict(self.theta, self.input_size, self.hidden_size, self.num_actions, state))
        return action
