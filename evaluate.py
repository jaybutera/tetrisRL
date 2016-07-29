from __future__ import print_function

import numpy as np

from keras.engine import Input, Model
from keras.layers import Dense, Flatten, Dropout

from engine import TetrisEngine
from models import Reinforcer, ReinforcementModel, fit_reinforcement

lr = 1e-2
penalty = 0.2


def sample(preds, temperature=1.0):
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


class TetrisReinforcer(Reinforcer):
    def __init__(self, engine):
        self.engine = engine
        self.width, self.height, self.nb_actions = engine.width, engine.height, engine.nb_actions

        # parameters to track cost
        self.prev_height = 0

    def get_maximum_height(self, board):
        for i in xrange(board.shape[1]):
            if np.any(board[:, i]):
                break
        else:
            return 0
        return board.shape[1] - i

    def get_nb_actions(self):
        return self.nb_actions

    def get_policy_gradients(self, distribution):
        action = sample(distribution.flatten(), temperature=0.5)

        params = self.engine.step(action)
        policy = np.zeros_like(distribution)

        height = self.get_maximum_height(params['board'])
        if params['died']:
            policy[action] = 10 * lr
            self.prev_height = 0
        else:
            policy[action] = (height - self.prev_height + penalty) * lr
            self.prev_height = height

        return policy

    def get_current_state(self):
        return self.engine.board


if __name__ == '__main__':
    width, height = 10, 20 # standard tetris friends rules
    batch_size = 20
    epoch_size = 100
    nb_epochs = 100

    engine = TetrisEngine(width, height)
    reinforcer = TetrisReinforcer(engine)

    input = Input(shape=(width, height), dtype='float32')
    flat = Flatten()
    dropout = Dropout(0.5)
    d1, d2 = Dense(100, activation='tanh', init='glorot_uniform'), Dense(engine.nb_actions, activation='softmax')
    output = d2(dropout(d1(flat(input))))
    model = ReinforcementModel(input=[input], output=[output])
    model.compile('sgd')

    # try training a regular model
    # model_normal = Model(input=[input], output=[output])
    # model_normal.compile('sgd', 'mse')
    # print(model_normal.predict([np.asarray([engine.board])]))

    fit_reinforcement(model, reinforcer, batch_size, epoch_size, nb_epochs)
