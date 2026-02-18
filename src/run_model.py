import curses
import sys
import os
import numpy as np
import torch
import time
from src.engine import TetrisEngine
from collections import deque
from src.dqn_agent import BasicFF, ReplayMemory, Transition, FRAME_STACK, make_stacked_state

# Pickle needs these classes visible in __main__ to unpickle checkpoints
import __main__
__main__.ReplayMemory = ReplayMemory
__main__.Transition = Transition

use_cuda = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

width, height = 10, 20 # standard tetris friends rules
engine = TetrisEngine(width, height)

def load_model(filename):
    model = BasicFF()
    if use_cuda:
        model.cuda()
    checkpoint = torch.load(filename, weights_only=False,
                            map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    return model

def run(model):
    first_frame = engine.clear()
    frames = deque([first_frame] * FRAME_STACK, maxlen=FRAME_STACK)
    state = make_stacked_state(frames)
    score = 0
    while True:
        with torch.no_grad():
            q_values = model(state.type(FloatTensor)).data
            action = q_values.max(1)[1].view(1,1).type(LongTensor)
            print(q_values)

        new_frame, reward, done = engine.step(action.item())
        frames.append(new_frame)
        state = make_stacked_state(frames)

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

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('specify a filename to load the model')
        sys.exit(1)

    filename = sys.argv[1]
    if os.path.isfile(filename):
        print("=> loading model '{}'".format(filename))
        model = load_model(filename).eval()
        run(model)
    else:
        print("=> no file found at '{}'".format(filename))
