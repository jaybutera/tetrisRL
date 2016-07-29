from __future__ import print_function

from keras import optimizers
from keras.engine.training import collect_trainable_weights
from keras.models import Model
import keras.backend as K

import numpy as np


class Reinforcer:
    def get_nb_actions(self):
        raise NotImplementedError()

    def get_policy_gradients(self, distribution):
        raise NotImplementedError()

    def get_current_state(self):
        raise NotImplementedError()


class ReinforcementModel(Model):
    def compile(self, optimizer, **kwargs):
        assert len(self.outputs) == 1, 'Model should have exactly one output:' + \
                                       'a probability distribution of actions to take'
        distribution = self.outputs[0]
        assert K.ndim(distribution) == 2, 'Model output should have dimensions <n_batches, n_actions>'

        # prepare targets of model
        shape = self.internal_output_shapes[0]
        name = self.output_names[0]
        self.targets = [K.placeholder(ndim=len(shape), name=name + '_target')]

        # compute loss with regularization
        total_loss = distribution
        for r in self.regularizers:
            total_loss = r(total_loss)

        # for training
        self.optimizer = optimizers.get(optimizer)
        self.total_loss = total_loss

        # functions for train, test and predict will
        # be compiled lazily when required.
        # This saves time when the user is not using all functions.
        self._function_kwargs = kwargs

        self.train_function = None
        self.test_function = None
        self.predict_function = None

    def _make_train_function(self):
        if not hasattr(self, 'train_function'):
            raise Exception('You must compile your model before using it.')
        if self.train_function is None:
            if self.uses_learning_phase and type(K.learning_phase()) is not int:
                inputs = self.inputs + self.targets + [K.learning_phase()]
            else:
                inputs = self.inputs + self.targets

            # get trainable weights
            trainable_weights = collect_trainable_weights(self)
            training_updates = self.optimizer.get_updates(trainable_weights, self.constraints,
                                                          K.sum(self.total_loss * self.targets[0]))
            updates = self.updates + training_updates

            self.train_function = K.function(inputs, [], updates=updates, **self._function_kwargs)

    def _make_test_function(self):
        raise Exception('Cannot call testing function on a reinforcement model')


def fit_reinforcement(model, reinforcer, batch_size, epoch_size, nb_epochs):
    model._make_train_function()
    model._make_predict_function()

    assert isinstance(reinforcer, Reinforcer)

    # basically the inputs and targets
    input_shape = reinforcer.get_current_state().shape
    policy_shape = reinforcer.get_nb_actions()
    states = np.zeros(shape=(batch_size,) + input_shape)
    policy_grads = np.zeros(shape=(batch_size, policy_shape))

    for epoch in range(nb_epochs):
        for batch in xrange(epoch_size):
            for i in xrange(batch_size):
                current_state = reinforcer.get_current_state()
                states[i] = current_state
                if model.uses_learning_phase and type(K.learning_phase()) is not int:
                    ins = [[current_state], 1.]
                else:
                    ins = [[current_state]]
                prediction = model.predict_function(ins)[0].flatten()
                policy_grads[i] = reinforcer.get_policy_gradients(prediction)
            if model.uses_learning_phase and type(K.learning_phase()) is not int:
                ins = [states, policy_grads, 1.]
            else:
                ins = [states, policy_grads]
            model.train_function(ins)
            print('\r%d / %d' % (batch, epoch_size), end='')
        print('\r%d / %d :: Deaths = %d' % (epoch, nb_epochs, reinforcer.engine.n_deaths))
