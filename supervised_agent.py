from dqn_agent import DQN
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch
from torch.autograd import Variable

FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor

def train(X_batch, Y_batch):
    optimizer.zero_grad()

    preds = model(X_batch)
    loss = criterion(preds, Y_batch)
    loss.backward()
    optimizer.step()

    return loss.data[0]

def save_checkpoint(state, filename):
    torch.save(state, filename)

#def load_model():
    #torch.load(

def load_dataset(filename):
    data = np.load('training_data.npy', allow_pickle=True)
    X_train = np.stack(data[:,0],
            axis=0).reshape((len(data),1,len(data[0][0]),len(data[0][0][0])))
    Y_train = data[:,3]

    print(X_train.shape)
    print(Y_train.shape)
    return X_train, Y_train

if __name__ == '__main__':
    model = DQN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Load data
    X_train, Y_train = load_dataset('training_data.npy')

    # Training loop
    for epoch in range(100):
        # Randomize and batch training data
        batchsize = 8
        # Randomly shuffle each epoch
        np.random.shuffle(X_train)
        np.random.shuffle(Y_train)
        # Batch
        X = np.array_split( X_train, batchsize) # States
        Y = np.array_split( Y_train, batchsize) # Actions

        loss = 0.
        for X_batch, Y_batch in zip(X,Y):
            X_batch = Variable(FloatTensor(X_batch), requires_grad=True)
            Y_batch = Variable(LongTensor(Y_batch), requires_grad=False)
            #Y_batch = Variable(LongTensor(Y_batch), raequires_grad=False)
            loss += train(X_batch, Y_batch)

        if epoch % 10 == 0:
            save_checkpoint({
                'epoch' : epoch,
                'best_score' : 0.,
                'state_dict' : model.state_dict()
                }, 'supervised_checkpoint.pth.tar')
            print('[{0}] loss: {1}'.format(epoch+1, loss))

