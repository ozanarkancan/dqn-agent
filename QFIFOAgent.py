import sys
import subprocess
from subprocess import Popen, PIPE
import numpy as np
# import cv2
from random import uniform, randint

class QAgent():
	def __init__(self, num_states, num_actions):
                self.num_states = num_states
                self.num_actions = num_actions
		self.Q = np.array([[0 for a in range(num_actions)] for s in range(num_states)])
		self.gamma = 1
                self.epsilon = 0.5
		self.previous_action = 0
		self.previous_state = 0
		
	def get_action(self,s):
                if (uniform(0,1) < self.epsilon):
                        best_action = randint(0,self.num_actions-1)
                else:
                        best_action = np.argmax(self.Q[s])
		self.previous_action = best_action
		self.previous_state = s
		return best_action
		
	def learn_from(self,s,r):
		self.Q[self.previous_state][self.previous_action] = r + self.gamma * self.Q[s][self.get_action(s)]
		
class Agent():
	def __init__(self, w, h):
		self.w = w
		self.h = h
	def get_action(self):
		return randint(0,17)

def get_screen_image(stream, w, h):
	img = np.array([])
	for i in xrange(h):
		row = stream[i*2*w:(i+1)*2*w]
		pixels = np.array([int(row[p*2:(p+1)*2],16) for p in range(w)])
		if len(img) == 0:
			img = pixels
		else:
			img = np.vstack((img, pixels))
	return img

if __name__== "__main__":
	ale = "/home/atilberk/Desktop/FALL2014/COMP408/Project/ale_0.4.4/ale_0_4/"
        num_states = 256
	agent = QAgent(num_states,18)

        for i in range(10):
                print "Episode",i+1
                p = Popen([ale + "ale", "-game_controller", "fifo", "-display_screen", "false", "-run_length_encoding", "false", ale + "roms/breakout.bin"], stdin=PIPE, stdout=PIPE)
                line = p.stdout.readline()
                w,h = map(int, line.split("-"))
                p.stdin.write("1,0,0,1\n")#screen and episode information
                total_reward = 0
                #cv2.namedWindow("screen", cv2.WINDOW_NORMAL)
	
                # first action
                line = p.stdout.readline()
                envinfo = line.strip()[:-1].split(":")
                img = get_screen_image(envinfo[0], w, h)
                state = int(np.average([255-x for x in [row[j] for j in range(w) for row in img]]))
                action = agent.get_action(state)
                #print "Action taken:",action
                p.stdin.write(str(action) + ",18\n")
	
                for line in iter(p.stdout.readline, b''):
                        envinfo = line.strip()[:-1].split(":")
                        terminal, reward = map(int, envinfo[1].split(","))
                        #print "Reward taken:",reward
                        total_reward += reward
                        #print "Total reward:",total_reward
                        if terminal == 1:
                                print "Total reward: ", total_reward
                                break
                        img = get_screen_image(envinfo[0], w, h)
                        state = int(np.average([255-x for x in [row[j] for j in range(w) for row in img if row[j] is not 250]]))
                        #print "New state:",state
                        agent.learn_from(state, reward)
                        action = agent.get_action(state)
                        #print "Action taken:",action
                        p.stdin.write(str(action) + ",18\n")	
