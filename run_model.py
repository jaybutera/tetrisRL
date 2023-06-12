import curses
import sys
import os
import torch
import time
from engine import TetrisEngine
from dqn_agent import BasicFF, ReplayMemory, Transition
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

width, height = 10, 20 # standard tetris friends rules
engine = TetrisEngine(width, height)

def load_model(filename):
    model = BasicFF()
    if use_cuda:
        model.cuda()
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])

    return model

def run(model):
    state = FloatTensor(engine.clear()[None,None,:,:])
    score = 0
    while True:
        action = model(Variable(state,
            volatile=True).type(FloatTensor)).data.max(1)[1].view(1,1).type(LongTensor)
        print( model(Variable(state,
            volatile=True).type(FloatTensor)).data)

        state, reward, done = engine.step(action.item())
        state = FloatTensor(state[None,None,:,:])

        # Accumulate reward
        score += int(reward)

        stdscr = curses.initscr()
        stdscr.clear()
        stdscr.addstr(str(engine))
        stdscr.addstr('\ncumulative reward: ' + str(score))
        stdscr.addstr('\nreward: ' + str(reward))
        time.sleep(.1)

        if done:
            print('score {0}'.format(score))
            break

if len(sys.argv) <= 1:
    print('specify a filename to load the model')
    sys.exit(1)

if __name__ == '__main__':
    filename = sys.argv[1]
    if os.path.isfile(filename):
        print("=> loading model '{}'".format(filename))
        model = load_model(filename).eval()
        run(model)
    else:
        print("=> no file found at '{}'".format(filename))
