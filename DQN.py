import sys
import subprocess
from subprocess import Popen, PIPE
import argparse
import numpy as np
import scipy.ndimage
from scipy.misc import imresize
import cv2
import theano
from random import uniform, randint
from nn import *
from memory import DataSet

class DQNAgent(object):
	def __init__(self, epsilon=0.5, memory=1000, batch_size=32):
		self.net = DeepNet()
		self.net.default_settings()
		self.epsilon = epsilon
		self.dataset = DataSet(length=memory)
		self.batch_size= batch_size
		self.gamma = 0.05
	
	def get_action(self, state):
		if self.epsilon >= np.random.random():
			a = np.random.randint(3)
		else:
			inpt = np.zeros((32, 84 * 84), dtype='float32')
			inpt[0,:] = state.flatten(2)
			out = self.net.predict(inpt)
			#out2 = self.net.single_predict(state.reshape(1,1,84,84))
			a = np.argmax(out[0])
		return a

	def update(self):
		print "Agent update"
		self.epsilon = np.max(self.epsilon - 0.001, 0.025)
		states, actions, rewards, terminal = self.dataset.get_random_batch()
		batch = np.zeros((self.batch_size, 4, 84, 84))
		y = np.zeros((self.batch_size, 3), dtype='float32')
		
		#Not stacked 4 frame
		#for i in xrange(3, self.batch_size + 3):
			#for j in xrange(4):
				#batch[i - 3, j, :, :] = states[i]

		q_vals = self.net.compute_q(states[:32, :,:].reshape(32, 84 * 84))
		
		for i in xrange(self.batch_size):
			if terminal[i]:
				y[i][action[i]] = rewards[i]
			else:
				a = np.argmax(q_vals[i,:])
				y[i][a] = rewards[i] + self.gamma * np.max(q_vals[i,:])
		
		#self.net.train_x.set_value((states[:32, :, :]).reshape(32, 84 * 84))
		#self.net.train_y.set_value(y)

		for epoch in range(1,11):
			loss = self.net.train_net(inp=states[:32, :, :].reshape(32, 84 * 84), target=y)
			print "Epoch: %i Loss: %f" % (epoch,loss)

def get_screen_image(stream, w, h):
	img = np.array([], dtype=theano.config.floatX)
	for i in xrange(h):
		row = stream[i*2*w:(i+1)*2*w]
                convert = lambda x : ((x/32) + ((x%32)/4) + ((x%4)*2))/24.0
		pixels = np.array([convert(int(row[p*2:(p+1)*2],16)) for p in range(w)])
		if len(img) == 0:
			img = pixels
		else:
			img = np.vstack((img, pixels))
	return img

def preprocess(img, ):
	img_down = imresize(img[34:194, :], 0.525) / 256.
        return img_down

def get_arg_parser():
	parser = argparse.ArgumentParser(prog="DQN")
	parser.add_argument("--ale", required=True, help="ale path")
	parser.add_argument("--rom", default="breakout.bin", help="rom name")
	parser.add_argument("--episodes", default=100, help="number of episodes")
	parser.add_argument("--updatetime", default=100, help="when the network must be trained (frame)")
	parser.add_argument("--n", default=1000, help="dataset memory size")
	parser.add_argument("--display", default="ever", help="'ever' for displaying ever, \
		'partial' for displaying at per 100 episode, 'never' for no display")
	return parser

if __name__== "__main__":
	theano.config.exception_verbosity = 'high'
	parser = get_arg_parser()
	parser.print_help()

	args = vars(parser.parse_args())
	ale = args['ale']
	rom = args['rom']
	episodes = args['episodes']
	updatetime = args['updatetime']
	n = args['n']

	agent = DQNAgent(memory=n)
	
	frame = 0

        for i in xrange(episodes):
                p = Popen([ale + "ale", "-game_controller", "fifo", "-display_screen", "true", "-run_length_encoding", "false", ale + "roms/" + rom], stdin=PIPE, stdout=PIPE)
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
                                print "Episode %i Total reward %i: ", (i + 1, total_reward)
				game_end = True
                        
			if frame != 0:
				agent.dataset.add_experience(previous_state, previous_action, reward, terminal == 1)

			if frame % updatetime == 0:
				agent.update()

			if game_end:
				break
                        
			action = agent.get_action(state)
                        previous_action = action
			previous_state = state
			if action > 0:
				action = action + 2
                        p.stdin.write(str(action) + ",18\n")
		p.kill()	
