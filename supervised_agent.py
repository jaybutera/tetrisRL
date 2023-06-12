from dqn_agent import DQN
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch

# Hyperparameters
batch_size = 32
# -----------

def train(X_batch, Y_batch):
    optimizer.zero_grad()

    preds = model(X_batch)
    loss = criterion(preds, Y_batch)
    loss.backward()
    optimizer.step()

    return loss.item()

def save_checkpoint(state, filename):
    torch.save(state, filename)

#def load_model():
    #torch.load(

def load_dataset(filename):
    data = np.load('training_data.npy', allow_pickle=True)
    X = np.stack(data[:,0],
            axis=0).reshape((len(data),1,len(data[0][0]),len(data[0][0][0]))) # States reshaped for CNN
    Y = data[:,3] # player's moves
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y.astype('uint8'), dtype=torch.uint8)

    n = int(len(X) * 0.9)
    X_train, Y_train = X[:n], Y[:n]
    X_val, Y_val = X[n:], Y[n:]

    return X_train, Y_train, X_val, Y_val

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    (x_data, y_data) = (X_train, Y_train) if split == 'train' else (X_val, Y_val)
    ix = torch.randint(len(x_data), (batch_size,))
    x = torch.stack([x_data[i] for i in ix])
    y = torch.stack([y_data[i] for i in ix])
    return x,y

if __name__ == '__main__':
    model = DQN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Load data
    X_train, Y_train, X_val, Y_val = load_dataset('training_data.npy')

    # Training loop
    for epoch in range(100):
        # Randomize and batch training data
        X_batch,Y_batch = get_batch('train')
        loss = 0.
        loss += train(X_batch, Y_batch)

        if epoch % 10 == 0:
            save_checkpoint({
                'epoch' : epoch,
                'best_score' : 0.,
                'state_dict' : model.state_dict()
                }, 'supervised_checkpoint.pth.tar')
            val_loss = criterion(model(X_val), Y_val).item()
            print('[{0}] train loss: {1:.2f} | val loss: {2:.2f}'.format(epoch+1, loss, val_loss))

