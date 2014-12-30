import sys
import subprocess
from subprocess import Popen, PIPE
import numpy as np
import scipy.ndimage
import cv2
from random import uniform, randint
from QNet import QNet

def get_screen_image(stream, w, h):
	img = np.array([])
	for i in xrange(h):
		row = stream[i*2*w:(i+1)*2*w]
                convert = lambda x : ((x/32) + ((x%32)/4) + ((x%4)*2))/24.0
		pixels = np.array([convert(int(row[p*2:(p+1)*2],16)) for p in range(w)])
		if len(img) == 0:
			img = pixels
		else:
			img = np.vstack((img, pixels))
	return img

def preprocess(img):
        img = scipy.ndimage.gaussian_filter(img, sigma=5)
        img_down = np.array([[row[x] for x in range(0,w,5)] for row in [img[y] for y in range(0,h,5)]])
        return img_down.reshape(np.size(img_down),1)

if __name__== "__main__":
	ale = "/home/atilberk/Desktop/FALL2014/COMP408/Project/ale_0.4.4/ale_0_4/"

	agent = QNet(21*16*4,18,64)

        for i in range(10):
                print "Episode",i+1
                p = Popen([ale + "ale", "-game_controller", "fifo", "-display_screen", "true", "-run_length_encoding", "false", ale + "roms/breakout.bin"], stdin=PIPE, stdout=PIPE)
                line = p.stdout.readline()
                w,h = map(int, line.split("-"))
                p.stdin.write("1,0,0,1\n")#screen and episode information
                total_reward = 0
                previous_action = 0
                #cv2.namedWindow("screen", cv2.WINDOW_NORMAL)
	
                # first action
                for line in iter(p.stdout.readline, b''):
                        envinfo = line.strip()[:-1].split(":")
                        img = get_screen_image(envinfo[0], w, h)
                        terminal, reward = map(int, envinfo[1].split(","))
                        total_reward += reward
                        if terminal == 1:
                                print "Total reward: ", total_reward
                                break
                        state = preprocess(img)
                        agent.train(state, previous_action, total_reward, terminal)
                        action = agent.get_action(state)
                        previous_action = action
                        p.stdin.write(str(action) + ",18\n")	
