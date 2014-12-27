import sys
import subprocess
from subprocess import Popen, PIPE
import numpy as np
import cv2
from random import randint

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
	ale = "/home/cano/cv/workspace/ale_0.4.4/ale_0_4/"

	p = Popen([ale + "ale", "-game_controller", "fifo", "-display_screen", "true", "-run_length_encoding", "false", ale + "roms/breakout.bin"], stdin=PIPE, stdout=PIPE)
	line = p.stdout.readline()
	w,h = map(int, line.split("-"))
	agent = Agent(w,h)
	p.stdin.write("1,0,0,1\n")#screen and episode information
	total_reward = 0
	#cv2.namedWindow("screen", cv2.WINDOW_NORMAL)
	for line in iter(p.stdout.readline, b''):
		envinfo = line.strip()[:-1].split(":")
		terminal, reward = map(int, envinfo[1].split(","))
		total_reward += reward
		if terminal == 1:
			print "Total reward: ", total_reward
			break
		action = agent.get_action()
		p.stdin.write(str(action) + ",18\n")
		#img = get_screen_image(envinfo[0], w, h)
		#print img
