import sys
import subprocess
from subprocess import Popen, PIPE
import numpy as np
# import scipy.ndimage
import cv2
from random import uniform, randint
from QNet import QNet

# converts the raw input stream into 2d grayscale numpy matrix of size h * w
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

# preprocesses 210 * 160 raw grayscale image matrix:
#  crops into 160 * 160
#  downsamples by 2
def preprocess(raw):
    img = raw[34:194]
    img_down = np.array([[row[x] for x in range(0,w,2)] for row in [img[y] for y in range(0,h,2)]])
    return img

def readopt(args):
    opts = []
    while len(args) > 0:
        if len(args) > 1 and args[1][0] != '-':
            opts.append((args[0],args[1]))
            args = args[2:]
        else:
            opts.append((args[0],''))
            args = args[1:]
    return opts

if __name__== "__main__":
        
    ale = "/home/atilberk/Desktop/FALL2014/COMP408/Project/ale_0.4.4/ale_0_4/"
    rom = "roms/pong.bin"
    display = "true"

    opts = readopt(sys.argv[1:])
    for opt, val in opts:
        if opt in ('-a', '--ale'):
            ale = val
        if opt in ('-r', '--rom'):
            rom = val
        if opt in ('-d', '--display-screen'):
            display = val.lower()

    D = []     #initialise experience set

    Q = QNet() # initilize the network


    for episode in range(10):
        print "Episode",episode+1

        # handshake
        p = Popen([ale + "ale", "-game_controller", "fifo", "-display_screen", display, "-run_length_encoding", "false", ale + rom], stdin=PIPE, stdout=PIPE)
        line = p.stdout.readline()
        w,h = map(int, line.split("-"))
        p.stdin.write("1,0,0,1\n")#screen and episode information

        # in-episode variables
        S = []
        A = []
        R = []
        X = []
        fi = []
        t = 0
        total_reward = 0

        # cv2.namedWindow("screen", cv2.WINDOW_NORMAL)

        # first input
        line = p.stdout.readline()
        envinfo = line.strip()[:-1].split(":")
        X.append(get_screen_image(envinfo[0], w, h))
        S.append(X[t])
        fi.append(S)

        for step in range(100000):
            # select an action
            A.append(Q.get_action(fi[t]))
            # execute the action
            p.stdin.write(str(action) + ",18\n")
            
            # observe the image and the reward
            line = p.stdout.readline()
            envinfo = line.strip()[:-1].split(":")
            X.append(get_screen_image(envinfo[0], w, h))
            terminal, reward = map(int, envinfo[1].split(","))
            R.append(reward)
            
            # set new state
            S.append(A[t])
            S.append(X[t+1])
            # preprocess the state
            fi.append(preprocess(S))
        
            # store the experience in D
            D.append((fi[t], A[t], R[t], fi[t+1]))
            if len(D) > 1000:
                D = D[len(D)-1000:]
            # pick random experience from D
            random_exp = D[randint(0,len(D)-1)]
            
            # train the network
            Q.train(random_exp, terminal)

            if terminal == 1:
                print "Total reward: ", sum(R)
                break

            t += 1
        p.kill()
