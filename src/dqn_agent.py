# -*- coding: utf-8 -*-
import math
import random
import sys
import os
import shutil
import numpy as np
#import matplotlib
#import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.engine import TetrisEngine

width, height = 10, 20 # standard tetris friends rules
engine = TetrisEngine(width, height)

# set up matplotlib
#is_ipython = 'inline' in matplotlib.get_backend()
#if is_ipython:
    #from IPython import display

#plt.ion()

# if gpu is to be used
use_cuda = torch.cuda.is_available()
if use_cuda:print("....Using Gpu...")
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
BoolTensor = torch.cuda.BoolTensor if use_cuda else torch.BoolTensor
#Tensor = FloatTensor


######################################################################
# Replay Memory
# -------------
# -  ``Transition`` - a named tuple representing a single transition in
#    our environment
# -  ``ReplayMemory`` - a cyclic buffer of bounded size that holds the
#    transitions observed recently. It also implements a ``.sample()``
#    method for selecting a random batch of transitions for training.
#

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.memory = []
        self.priorities = []
        self.position = 0
        self.alpha = alpha

    def push(self, *args):
        """Saves a transition."""
        max_prio = max(self.priorities) if self.priorities else 1.0
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.priorities.append(max_prio)
        self.memory[self.position] = Transition(*args)
        self.priorities[self.position] = max_prio
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        prios = np.array(self.priorities, dtype=np.float64)
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.memory), batch_size, p=probs, replace=False)
        # Importance-sampling weights
        weights = (len(self.memory) * probs[indices]) ** (-beta)
        weights /= weights.max()
        samples = [self.memory[i] for i in indices]
        return samples, indices, FloatTensor(weights)

    def update_priorities(self, indices, td_errors):
        for idx, td in zip(indices, td_errors):
            self.priorities[idx] = abs(td) + 1e-6

    def __len__(self):
        return len(self.memory)


'''
class DQN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        #self.conv3 = nn.Conv2d(32, 32, kernel_size=2, stride=2)
        #self.bn3 = nn.BatchNorm2d(32)
        #self.rnn = nn.LSTM(448, 240)
        self.lin1 = nn.Linear(768, 256)
        self.head = nn.Linear(256, engine.nb_actions)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        #x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.lin1(x.view(x.size(0), -1)))
        return self.head(x.view(x.size(0), -1))
'''

class BasicFF(nn.Module):
    def __init__(self, fan_in=200, fan_out=7):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(3200, 256)
        self.fc2 = nn.Linear(256, fan_out)

    def forward(self, x):
        if len(x.shape) == 2:
            # Flat input (200,) -> (1, 10, 20)
            x = x.view(x.shape[0], 1, 10, 20)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
       
######################################################################
# Training
# --------
#
# Hyperparameters and utilities
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This cell instantiates our model and its optimizer, and defines some
# utilities:
#
# -  ``Variable`` - this is a simple wrapper around
#    ``torch.autograd.Variable`` that will automatically send the data to
#    the GPU every time we construct a Variable.
# -  ``select_action`` - will select an action accordingly to an epsilon
#    greedy policy. Simply put, we'll sometimes use our model for choosing
#    the action, and sometimes we'll just sample one uniformly. The
#    probability of choosing a random action will start at ``EPS_START``
#    and will decay exponentially towards ``EPS_END``. ``EPS_DECAY``
#    controls the rate of the decay.
#

BATCH_SIZE = 64
GAMMA = 0.8
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 5000
TARGET_UPDATE = 50
TRAIN_STEPS = 4
REWARD_SCALE = 0.01
MEMORY_SIZE = 50000
CHECKPOINT_FILE = 'checkpoint.pth.tar'
steps_done = 0


@torch.no_grad()
def select_action(state, steps_done):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        actions = model(state.type(FloatTensor))
        return actions.data.max(1)[1].view(1, 1)
    else:
        return FloatTensor([[random.randrange(engine.nb_actions)]])


episode_durations = []


'''
def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(episode_durations)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())
'''


######################################################################
# Training loop
# ^^^^^^^^^^^^^
#
# Finally, the code for training our model.
#
# Here, you can find an ``optimize_model`` function that performs a
# single step of the optimization. It first samples a batch, concatenates
# all the tensors into a single one, computes :math:`Q(s_t, a_t)` and
# :math:`V(s_{t+1}) = \max_a Q(s_{t+1}, a)`, and combines them into our
# loss. By defition we set :math:`V(s) = 0` if :math:`s` is a terminal
# state.


last_sync = 0


def optimize_model():
    global last_sync
    if len(memory) < BATCH_SIZE:
        return
    transitions, indices, weights = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=torch.bool)

    with torch.no_grad():
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = model(state_batch).gather(1, action_batch.long()).squeeze()

    with torch.no_grad():
        next_state_values = torch.zeros(BATCH_SIZE).type(FloatTensor)
        next_state_values[non_final_mask] = target_model(non_final_next_states).max(1)[0]
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Per-element loss weighted by importance sampling
    td_errors = state_action_values - expected_state_action_values
    loss = (weights * F.smooth_l1_loss(state_action_values, expected_state_action_values, reduction='none')).mean()

    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    # Update replay priorities
    memory.update_priorities(indices, td_errors.detach().cpu().numpy())

    return loss.item()

def optimize_supervised(pred, targ):
    optimizer.zero_grad()

    diff = loss(pred, targ)
    diff.backward()
    optimizer.step()

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def load_checkpoint(filename):
    checkpoint = torch.load(filename, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    try: # If these fail, its loading a supervised model
        optimizer.load_state_dict(checkpoint['optimizer'])
        memory = checkpoint['memory']
    except Exception as e:
        pass
    # Low chance of random action
    #steps_done = 10 * EPS_DECAY

    return checkpoint['epoch'], checkpoint['best_score']

if __name__ == '__main__':
    model = BasicFF()
    target_model = BasicFF()
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    print(model)

    if use_cuda:
        model.cuda()
        target_model.cuda()

    loss = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=1e-4)
    memory = ReplayMemory(MEMORY_SIZE)

    # Check if user specified to resume from a checkpoint
    start_epoch = 0
    best_score = 0
    if len(sys.argv) > 1 and sys.argv[1] == 'resume':
        if len(sys.argv) > 2:
            CHECKPOINT_FILE = sys.argv[2]
        if os.path.isfile(CHECKPOINT_FILE):
            print("=> loading checkpoint '{}'".format(CHECKPOINT_FILE))
            start_epoch, best_score = load_checkpoint(CHECKPOINT_FILE)
            target_model.load_state_dict(model.state_dict())
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(CHECKPOINT_FILE, start_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(CHECKPOINT_FILE))

    ######################################################################
    #
    # Below, you can find the main training loop. At the beginning we reset
    # the environment and initialize the ``state`` variable. Then, we sample
    # an action, execute it, observe the next screen and the reward (always
    # 1), and optimize our model once. When the episode ends (our model
    # fails), we restart the loop.

    steps_done = 0
    f = open('log.out', 'w+')
    for i_episode in count(start_epoch):
        # Initialize the environment and state
        state = FloatTensor(engine.clear()[None,None,:,:])

        score = 0
        for t in count():
            # Select and perform an action
            action = select_action(state, steps_done).type(LongTensor)

            # Observations
            last_state = state
            state, reward, done = engine.step(action.item())
            state = FloatTensor(state[None,None,:,:])

            # Accumulate reward
            score += int(reward)

            reward = FloatTensor([float(reward) * REWARD_SCALE])
            # Store the transition in memory
            memory.push(last_state, action, state, reward)

            # Perform one step of the optimization (on the target network)
            if done:
                # Train model - multiple steps per episode
                log = 'epoch {0} score {1}'.format(i_episode, score)
                f.write(log + '\n')
                loss_val = None
                for _ in range(TRAIN_STEPS):
                    loss_val = optimize_model()
                if loss_val and i_episode % 100 == 0:
                    print(f'[{i_episode}] loss: {loss_val:.2f} score: {score}')
                # Update target network
                if i_episode % TARGET_UPDATE == 0:
                    target_model.load_state_dict(model.state_dict())
                # Checkpoint
                if i_episode % 100 == 0:
                    is_best = True if score > best_score else False
                    save_checkpoint({
                        'epoch' : i_episode,
                        'state_dict' : model.state_dict(),
                        'best_score' : best_score,
                        'optimizer' : optimizer.state_dict(),
                        'memory' : memory
                        }, is_best)
                break

    f.close()
    print('Complete')
