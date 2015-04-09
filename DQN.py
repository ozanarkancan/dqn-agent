import sys
import subprocess
from subprocess import Popen, PIPE
import argparse
import numpy as np
import scipy.ndimage
from PIL import Image
import theano
from random import uniform, randint
from nn import *
from memory import DataSet
from romsettings import *
import color
import cPickle

class DQNAgent(object):
	def __init__(self, n_actions=18, epsilon=1, memory=5000, batch_size=32):
		self.net = DeepNet()
		self.net.default_settings(n_actions)
		self.epsilon = epsilon
		self.dataset = DataSet(length=memory)
		self.batch_size= batch_size
		self.gamma = 0.95
		self.n_actions = n_actions
	
	def get_action(self, state):
		if self.epsilon >= np.random.random():
			a = np.random.randint(self.n_actions)
		else:
			frames = self.dataset.get_stacked_frames()
			inpt = np.zeros((32, 4 * 84 * 84), dtype='float32')

			inpt[0,:] = np.concatenate((state.flatten(2), frames.flatten(2)))
			out = self.net.predict(inpt)
			#print out
			a = out[0]
		return a

	def update(self):
		#print "Agent update"
		self.epsilon = np.max(self.epsilon - 0.0001, 0.05)
		states, actions, rewards, terminals = self.dataset.get_random_batch()
		batch = np.zeros((self.batch_size, 4, 84, 84), dtype='float32')
		y = np.zeros((self.batch_size, self.n_actions), dtype='float32')
		for i in xrange(3, self.batch_size + 3):
			for j in xrange(4):
				batch[i - 3, j, :, :] = states[i]

		#q_vals = self.net.compute_q(states[:32, :,:].reshape(32, 84 * 84))
		q_vals = self.net.compute_q(batch.reshape(32, 4 * 84 * 84))
		#print "Qs: ", q_vals
		for i in xrange(self.batch_size):
			if terminals[i]:
				y[i][actions[i]] = rewards[i]
			else:
				a = np.argmax(q_vals[i,:])
				y[i][a] = rewards[i] + self.gamma * np.max(q_vals[i,:])
		
		#print "Batch:\n", batch[:,:, 20:70, 20:70]
		self.net.train_x.set_value(batch.reshape(32, 4 * 84 * 84))
		self.net.train_y.set_value(y)
		
		#print self.net.layers[0].input.eval()
		#print "Weights: "
		#for l in self.net.layers:
		#	print l.params[0].get_value()

		for epoch in range(1,11):
			loss = self.net.train_net()
			#print "Epoch: %i Loss: %f" % (epoch,loss)
	def save(self):
		f = file(os.path.realpath('.') + '/atari/rnn-agent/save/net.save', 'wb')
		for p in self.net.params:
			cPickle.dump(p.get_value(), f, protocol=cPickle.HIGHEST_PROTOCOL)
		f.close()
	
	def load(self):
		f = file(os.path.realpath('.') + '/atari/rnn-agent/save/net.save', 'rb')
		for i in range(len(self.net.params)):
			self.net.params[i].set_value(cPickle.load(f))
		f.close()
	
	def train(self):
		frame = 0
		total_rewards = 0.

        	for i in xrange(episodes):
                	p = Popen([ale + "ale", "-game_controller", "fifo", "-display_screen", "true", "-frame_skip", "3", "-run_length_encoding", "false", ale + "roms/" + rom], stdin=PIPE, stdout=PIPE, stderr=PIPE)
                	line = p.stdout.readline()
                	w,h = map(int, line.split("-"))
                	p.stdin.write("1,0,0,1\n")#screen and episode information
                	total_reward = 0
                	previous_action = 0           
			previous_state = None

                	#game loop
			game_end = False
                	for line in iter(p.stdout.readline, b''):
				frame += 1
                        	envinfo = line.strip()[:-1].split(":")
                        	img = get_screen_image(envinfo[0], w, h)
                        	terminal, reward = map(int, envinfo[1].split(","))
                        	total_reward += reward
                        	state = preprocess(img)
				if terminal == 1:
                                	print "Episode: %i Total reward: %i " % (i + 1, total_reward)
					total_rewards += total_reward
					game_end = True
                        
				if not previous_state is None:
					agent.dataset.add_experience(previous_state, previous_action, reward, terminal == 1)

				if frame % updatetime == 0 and agent.dataset.available > 100:
					agent.update()

				if game_end:
					break
                        
				action = agent.get_action(state)

                        	previous_action = action
				previous_state = state
				action = map_action(action, rom)
                        	p.stdin.write(str(action) + ",18\n")
			p.kill()
			self.save()
	
		print "Average Total Rewards: %f" % (total_rewards / episodes)

	def play(self):
		self.load()
		self.epsilon = 0.05
		for i in xrange(episodes):
                	p = Popen([ale + "ale", "-game_controller", "fifo", "-display_screen", "true", "-frame_skip", "3", "-run_length_encoding", "false", ale + "roms/" + rom], stdin=PIPE, stdout=PIPE, stderr=PIPE)
                	line = p.stdout.readline()
                	w,h = map(int, line.split("-"))
                	p.stdin.write("1,0,0,1\n")#screen and episode information
                	total_reward = 0

                	#game loop
			game_end = False
                	for line in iter(p.stdout.readline, b''):
                        	envinfo = line.strip()[:-1].split(":")
                        	img = get_screen_image(envinfo[0], w, h)
                        	terminal, reward = map(int, envinfo[1].split(","))
				total_reward += reward
                        	state = preprocess(img)
				if terminal == 1:
                                	print "Episode: %i Total reward: %i " % (i + 1, total_reward)
					game_end = True
					break
                        
				action = agent.get_action(state)
				action = map_action(action, rom)
                        	p.stdin.write(str(action) + ",18\n")
			p.kill()

def get_screen_image(stream, w, h):
	img = np.array([], dtype=theano.config.floatX)
	for i in xrange(h):
		row = stream[i*2*w:(i+1)*2*w]
		pixels = np.array([color.Palette[row[p*2:(p+1)*2]] for p in range(w)])
		if len(img) == 0:
			img = pixels
		else:
			img = np.vstack((img, pixels))
	return img

def preprocess(img, ):
	img_down = Image.fromarray(img[33:193,:])
	img_down.thumbnail((84, 84), Image.NEAREST)
	img_down = np.asarray(img_down,dtype='float32')
        return img_down

def get_arg_parser():
	parser = argparse.ArgumentParser(prog="DQN")
	parser.add_argument("--ale", required=True, help="ale path")
	parser.add_argument("--rom", default="breakout.bin", help="rom name")
	parser.add_argument("--episodes", default=200, type=int, help="number of episodes")
	parser.add_argument("--updatetime", default=5, type=int, help="when the network must be trained (frame)")
	parser.add_argument("--n", default=10000, type=int, help="dataset memory size")
	parser.add_argument("--display", default="ever", help="'ever' for displaying ever, \
		'partial' for displaying at per 100 episode, 'never' for no display")
	parser.add_argument("--mode", default="train", help="train or play")
	return parser

if __name__== "__main__":
	theano.config.exception_verbosity = 'high'
	parser = get_arg_parser()

	args = vars(parser.parse_args())
	ale = args['ale']
	rom = args['rom']
	episodes = args['episodes']
	updatetime = args['updatetime']
	n = args['n']
	mode = args['mode']

	agent = DQNAgent(
		n_actions=get_number_of_legal_actions(rom),
		memory=n)
	
	if mode == "train":
		agent.train()
	else:
		agent.play()
