import click
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from engine import TetrisEngine


class PolicyNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(PolicyNetwork, self).__init__()
        self.input_dims = input_dims
        self.lr = lr
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu:0')

        self.to(self.device)

    def forward(self, observation):
        float_np = observation.reshape(self.input_dims).astype(np.float32)
        state = T.tensor(float_np).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Agent(object):
    def __init__(self, lr, input_dims, gamma=0.99, n_actions=4, l1_size=256, l2_size=256):
        self.gamma = gamma
        self.reward_memory = []
        self.action_memory = []
        self.policy = PolicyNetwork(lr, input_dims, l1_size, l2_size, n_actions)

    def choose_action(self, observation):
        probabilities = F.softmax(self.policy.forward(observation))
        action_prob = T.distributions.Categorical(probabilities)
        action = action_prob.sample()
        log_probs = action_prob.log_prob(action)
        self.action_memory.append(log_probs)
        return action.item(), probabilities

    def store_rewards(self, reward):
        self.reward_memory.append(reward)

    def clear(self):
        self.action_memory = []
        self.reward_memory = []

    def learn(self):
        self.policy.optimizer.zero_grad()
        G = np.zeros_like(self.reward_memory, dtype=np.float64)
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum += self.reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        G = (G - mean) / std
        G = T.tensor(G, dtype=T.float).to(self.policy.device)
        loss = 0
        for g, lobprob in zip(G, self.action_memory):
            loss += -g * lobprob
        loss.backward()
        self.policy.optimizer.step()
        self.action_memory = []
        self.reward_memory = []


@click.command()
@click.option('--episode', default=5)
@click.option('--load/--no-load', default=True)
@click.option('--learn/--no-learn', default=False)
@click.option('--debug/--no-debug', default=False)
@click.option('--random_rate', default=0.0)
@click.option('--session', default="")
def main(episode, load, learn, debug, random_rate, session):
    load_model = load
    print("load model", load_model, "learn", learn, "debug", debug, "episode", episode)


    width, height = 10, 20 # standard tetris friends rules
    env = TetrisEngine(width, height)
    action_count = 7
    agent = Agent(lr=0.001, input_dims=width*height, gamma=0.99, n_actions=action_count, l1_size=128, l2_size=128)
    if session:
        model_filename = "%s-trained_model.torch" % session
    else:
        model_filename = "trained_model.torch"
    parameter_size = sum([len(p) for p in agent.policy.parameters()])
    print("network parameter size:", parameter_size)

    if load_model:
        agent.policy.load_state_dict(T.load(model_filename))
    score_history = []
    for i in range(episode):
        done = False
        score = 0
        state = env.clear()
        counter = 0
        while not done:
            counter += 1
            if random.random() < random_rate:
                action = random.randint(0, len(actions) - 1)
                prob = -1
            else:
                action, probs = agent.choose_action(state)
                prob = probs[action].item()
            state, reward, done = env.step(action)
            agent.store_rewards(reward)
            score += reward
            if i % 100 == 0 and counter % 100 == 1:
                print('episode: ', i, 'counter: ', counter, 'reward %0.3f' % reward, 'action: %s (%0.2f)' % (action, prob))
        if i % 100 == 0:
            print('episode: ', i, 'score %0.3f' % score)
        score_history.append(score)
        if learn:
            if score > 0:
                agent.learn()
                if i % 100 == 0:
                    T.save(agent.policy.state_dict(), model_filename)
            else:
                agent.clear()
if __name__ == '__main__':
    main()
