import torch.nn as nn
import torch.nn.functional as F
from Environment import *

def matrix_to_vector(matrix):
    list = []
    for i in matrix:
        for ii in matrix[i]:
            list.append(matrix[i][ii])
    return list

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(8 + 5, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 20)

    def forward(self, s0, s1):
        x = torch.cat((s0, s1), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        readout = self.fc3(x)
        return readout

    def init(self):
        self.fc1.weight.data = torch.abs(0.01 * torch.randn(self.fc1.weight.size()))
        self.fc2.weight.data = torch.abs(0.01 * torch.randn(self.fc2.weight.size()))
        self.fc3.weight.data = torch.abs(0.01 * torch.randn(self.fc3.weight.size()))

        self.fc1.bias.data = torch.ones(self.fc1.bias.size()) * 0.01
        self.fc2.bias.data = torch.ones(self.fc2.bias.size()) * 0.01
        self.fc3.bias.data = torch.ones(self.fc3.bias.size()) * 0.01


import numpy as np
import torch
import pandas as pd

ACTIONS = 20
GAMMA = 0.99
INITIAL_EPSILON = 0.6
FINAL_EPSILON = 0.001
OBSERVE = xx
REPLAY_MOMERY = xx
BATCH = 400
EXPLORE = xx
TRAIN = xxx

net = Net()
net.init()
# net.cuda()
criterion = nn.MSELoss() #.cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-6)

Offload = stepgo(301, 6)
D = []

s_t = Offload.pickState()
s_t[0] = list(np.divide(s_t[0], 301))
s_t[1][0] = s_t[1][0]/8
s_t[1][1] = s_t[1][1]/100
s_t[1][2] = s_t[1][2]/100
s_t[1][3] = s_t[1][3]/1100 # sum_mMTC
s_t[1][4] = s_t[1][4]/1600 # sum_uRLLC
Vect = 0

epsilon = INITIAL_EPSILON
timer = 0
state = ''
loss_value = []
reward = []
time_slot_l = []
time_slot_r = []



while timer < (OBSERVE + EXPLORE + TRAIN):
    s0 = torch.FloatTensor(s_t[0]).view(-1, 8)
    s1 = torch.FloatTensor(s_t[1]).view(-1, 5)
    readout = net(s0, s1)
    readout_t = readout.data.numpy()

    a_t = list(np.zeros(20))
    if random.random() <= epsilon:
        action_index = random.randrange(ACTIONS)
    else:
        action_index = np.argmax(readout_t)
    a_t[action_index] = 1

    if epsilon > FINAL_EPSILON and timer > OBSERVE:
        epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

    s_t1, r_t = Offload.ActionToReward(a_t, timer, Vect)

    s_t1[0] = list(np.divide(s_t1[0], 301))
    s_t1[1][0] = s_t1[1][0] / 8
    s_t1[1][1] = s_t1[1][1] / 100
    s_t1[1][2] = s_t1[1][2] / 100
    s_t1[1][3] = s_t1[1][3] / 1100
    s_t1[1][4] = s_t1[1][4] / 1600


    D.append([s_t, a_t, r_t, s_t1])
    if len(D) > REPLAY_MOMERY:
        D = D[1:]

    if timer > OBSERVE:
        starter = random.randrange(0, REPLAY_MOMERY - BATCH)
        minibatch = D[starter:(starter + BATCH)]
        optimizer.zero_grad()

        s0_j_batch = list([d[0][0] for d in minibatch])
        s1_j_batch = list([d[0][1] for d in minibatch])

        a_batch = list([d[1] for d in minibatch])
        r_batch = list([d[2] for d in minibatch])

        s0_j1_batch = list([d[3][0] for d in minibatch])
        s1_j1_batch = list([d[3][1] for d in minibatch])

        s10 = torch.FloatTensor(s0_j1_batch).view(-1, 8)
        s11 = torch.FloatTensor(s1_j1_batch).view(-1, 5)

        readout1 = net(s10, s11)
        readout_j1_batch = readout1.data.numpy()
        #
        y_batch = []
        for i in range(0, len(minibatch)):
            y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))
        y = torch.from_numpy(np.array(y_batch, dtype=float))
        a = torch.from_numpy(np.array(a_batch, dtype=float))

        s00 = torch.FloatTensor(s0_j_batch).view(-1, 8)
        s01 = torch.FloatTensor(s1_j_batch).view(-1, 5)

        readout0 = net(s00, s01)
        readout_action = readout0.mul(a).sum(1)
        loss = criterion(readout_action, y)
        loss.backward()
        optimizer.step()
        if timer % 500 == 0:
            loss = loss.detach().numpy()
            print('loss', loss)
            loss_value.append(loss)
            time_slot_l.append(timer)
    timer += 1

    if Vect < 7:
        Vect += 1
    else:
        Vect = 0

    s_t = Offload.NetStateUpdate(timer, Vect)
    s_t[0] = list(np.divide(s_t[0], 301))
    s_t[1][0] = s_t[1][0]/8
    s_t[1][1] = s_t[1][1]/100
    s_t[1][2] = s_t[1][2]/100
    s_t[1][3] = s_t[1][3]/1100
    s_t[1][4] = s_t[1][4]/1600

    gotta = {'net': net.state_dict(), 'optimizer':optimizer.state_dict()}

    if timer <= OBSERVE:
        state = 'observe'
    elif timer > OBSERVE and timer <= OBSERVE + EXPLORE:
        state = 'explore'
    else:
        state = 'train'
    if timer % 500 == 0:
        sss = 'time_step {}/ state {}/ Epsilon {:.2f}/ action {}/ reward {}/ Q_MAX {:e}/'.format(
            timer, state, epsilon, action_index, r_t, np.max(readout_t)
        )
        print(sss)
        f = open('', 'a')
        f.write(sss + '\n')
        f.close()

    if timer % 5 == 0:
        time_slot_r.append(timer)
        reward.append(r_t)

    if timer % 500 == 0:
        data_loss = {'loss': loss_value,
                     'time_l': time_slot_l}
        data_reward = {'reward': reward,
                       'time_r': time_slot_r}

        data_loss = pd.DataFrame(data_loss)
        data_loss.to_csv('', index=False)
        data_reward = pd.DataFrame(data_reward)
        data_reward.to_csv('')



