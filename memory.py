import numpy as np
import theano

floatX = theano.config.floatX

class DataSet(object):
	def __init__(self, length=1000, img_size=(84, 84)):
		self.count = 0
		self.length = length
		self.states = np.zeros((length, img_size[0], img_size[1]), dtype=floatX)
		self.actions = np.zeros(length, dtype='int32')
		self.rewards = np.zeros(length, dtype=floatX)
		self.terminals = np.zeros(length, dtype='bool')
		self.available = 0
	
	def add_experience(self, state, action, reward, terminal):
		self.states[self.count, :, :] = state
		self.actions[self.count] = action
		self.rewards[self.count] = reward
		self.terminals[self.count] = terminal
		self.count += 1
		
		if self.available != self.length:
			self.available = self.count

		if self.count == self.length:
			self.count = 0
	
	def get_stacked_frames(self):
		indices = map(lambda x: x % self.length, range(self.count - 3, self.count))
		return self.states[indices, : , :]


	def get_random_batch(self, batch_size=32):
		indx = np.random.randint(self.available)
		indices = (range(self.length) + range(35))[indx:(indx + batch_size + 3)]
		return self.states[indices, :, :], self.actions[indices], self.rewards[indices], self.terminals[indices]
