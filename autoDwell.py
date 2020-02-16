import math
import random
import numpy as np
from collections import namedtuple

from obj.env import Env

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from models.gat import GAT

# if gpu is to be used
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

#policy
class Policy(nn.Module):


    def __init__(self):
        super(Policy, self).__init__()
        # self.action_space = env.action_space.n
        
        self.rnn_flow = nn.GRU(Env.FLOW_SIZE, Env.RNN_HIDDEN, n_layers = 1, dropout=Env.DROPOUT_RATE)
        self.gat = GAT(nfeat=Env.FLOW_SIZE, nhid = Env.GAT_HIDDEN, nouput = Env.GAT_OUTPUT, dropout = 0.6, nheads = 4, alpha = 0.2)
        self.rnn_conclusive = nn.GRU(Env.FLOW_SIZE, Env.RNN_HIDDEN, n_layers = 1, dropout=Env.DROPOUT_RATE)
        self.ext_feature_embed_l1 = nn.Linear(Env.EXT_FEATURE_SIZE, Env.EXT_FEATURE_EMBED_HIDDEN, bias=False)
        self.ext_feature_embed_l2 = nn.Linear(Env.EXT_FEATURE_EMBED_HIDDEN, Env.FEATURE_EMBED_OUTPUT, bias=False)

        self.tra_feature_embed_l1 = nn.Linear(Env.TRAIN_FEATURE_SIZE, Env.TRAIN_FEATURE_EMBED_HIDDEN, bias=False)
        self.tra_feature_embed_l2 = nn.Linear(Env.TRAIN_FEATURE_EMBED_HIDDEN, Env.FEATURE_EMBED_OUTPUT, bias=False)

        self.v_front = nn.Linear(Env.FEATURE_EMBED_OUTPUT*2, 1, bias=False)
        self.v_rear = nn.Linear(Env.FEATURE_EMBED_OUTPUT*2, 1, bias=False)
        self.softmax = nn.Softmax(1)

        self.fusion_l1 = nn.Linear(Env.FEATURE_EMBED_OUTPUT*3+Env.RNN_HIDDEN, Env.FUSION_HIDDEN, bias=False)
        self.fusion_l2 = nn.Linear(Env.FUSION_HIDDEN, Env.ACTION_NUM, bias=False)

        self.dropout = nn.Dropout(p=Env.DROPOUT_RATE)
        self.activation = nn.ReLU()

        self.steps_done = 0
        self.episode_num = 0
        self.last_update = 0
        self.losses = []

    #############################################
    # x_flows is a array with passenger flow vectors of the station and its neighbors (if the station is a transfer learner)
    # x_features is a array with extend feature vector of the station and its neighbors (if the station is a transfer learner)
    # x_flow_adjs is a array with the adjacency matrix of the station (if the station is a transfer learner, otherwise -1)
    # x_flow_adjs is a array with the indexces denoting the current station in the corresponding adjacency matrix (if the station is a transfer learner, otherwise -1)
    # x_train is the feature vector of the current train
    # x_trains_context_front is a tensor representing features of front trains of the current train
    # x_trains_context_rear is a tensor representing features of rear trains of the current train

    def forward(self, x_flows, x_features, x_flow_adjs, indexes, x_train, x_trains_context_front, x_trains_context_rear):
        # passenger feature extractor
        x_stations = torch.empty((1, Env.FEATURE_EMBED_OUTPUT+Env.RNN_HIDDEN)).to(Env.device)
        for i in range(len(x_flows)):
            x_flow = x_flows[i]
            x_flow_adj = x_flow_adjs[i]
            x_feature = x_features[i]
            
            #embed features
            x_feature = self.ext_feature_embed_l1(x_feature)
            x_feature = self.activation(x_feature)
            x_feature = self.ext_feature_embed_l2(x_feature)
            
            #embed flows
            x_flow = x_flow.view(-1, Env.FLOW_SIZE, 1)
            _, x_flow = self.flow(x_flow)
            x_flow = x_flow.squeeze()

            if x_flow_adj == -1:
                x_flow = self.gat(x_flows, x_flow_adj, indexes[i])
                x_flow = x_flows.view(1, -1)
            
            x_station = torch.cat((x_feature, x_flow), dim=1)
            x_stations = torch.cat((x_station, x_stations))

        x_stations = x_stations.view(-1, len(x_flows), Env.FEATURE_EMBED_OUTPUT+Env.RNN_HIDDEN)
        x_stations = self.rnn_conclusive(x_stations)

        # Train feature extractor

        x_train = self.tra_feature_embed_l1(x_train)
        x_train = self.activation(x_train)
        x_train = self.tra_feature_embed_l2(x_train)
        if x_trains_front is not None:
            x_trains_front = self.tra_feature_embed_l1(x_trains_front)
            x_trains_front = self.activation(x_trains_front)
            x_trains_front = self.tra_feature_embed_l2(x_trains_front)

            x_train_concat = torch.cat((x_trains_front, x_train.expend(x_trains_front.size(0), -1)), dim=1)
            atten_front = self.v_front(x_train_concat)
            atten_front = self.softmax(atten_front)
            x_trains_front_output = torch.bmm(atten_front, x_trains_front)
            x_trains_front_output = x_trains_front_output.sum(1)/x_trains_front.size(0)

        if x_trains_rear is not None:
            x_trains_rear = self.tra_feature_embed_l1(x_trains_rear)
            x_trains_rear = self.activation(x_trains_rear)
            x_trains_rear = self.tra_feature_embed_l2(x_trains_rear)

            x_train_concat = torch.cat((x_trains_rear, x_train.expend(x_trains_rear.size(0), -1)), dim=1)
            atten_rear = self.v_rear(x_train_concat)
            atten_rear = self.softmax(atten_rear)
            x_trains_rear_output = torch.bmm(atten_rear, x_trains_rear)
            x_trains_rear_output = x_trains_rear_output.sum(1)/x_trains_rear.size(0)

        x_trains = torch.cat((x_train, x_trains_context_front, x_trains_rear_output), dim=1)

        #fusion component
        x = torch.cat((x_stations, x_trains), dim=1)
        x = self.fusion_l1(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.fusion_l2(x)

        return x

# Training
# --------
# 
# Hyperparameters and utilities
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This cell instantiates our model and its optimizer, and defines some
# utilities:
# 
# -  ``select_action`` - will select an action accordingly to an epsilon
#    greedy policy. Simply put, we'll sometimes use our model for choosing
#    the action, and sometimes we'll just sample one uniformly. The
#    probability of choosing a random action will start at ``EPS_START``
#    and will decay exponentially towards ``EPS_END``. ``EPS_DECAY``
#    controls the rate of the decay.
# -  ``plot_durations`` - a helper for plotting the durations of episodes,
#    along with an average over the last 100 episodes (the measure used in
#    the official evaluations). The plot will be underneath the cell
#    containing the main training loop, and will update after every
#    episode.
# 

def select_action(x_flows, x_features, x_flow_adjs, indexes, x_train, x_trains_context_front, x_trains_context_rear, policy_net):
    sample = random.random()
    eps_threshold = Env.EPS_END + (Env.EPS_START - Env.EPS_END) * math.exp(-1. * policy_net.steps_done / Env.EPS_DECAY)
    policy_net.steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(x_flows, x_features, x_flow_adjs, indexes, x_train, x_trains_context_front, x_trains_context_rear).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(Env.ACTION_NUM)]],  device=Env.device, dtype=torch.long)


# Training loop
# ^^^^^^^^^^^^^
# 
# Finally, the code for training our model.
# 
# Here, you can find an ``optimize_model`` function that performs a
# single step of the optimization. It first samples a batch, concatenates
# all the tensors into a single one, computes $Q(s_t, a_t)$ and
# $V(s_{t+1}) = \max_a Q(s_{t+1}, a)$, and combines them into our
# loss. By defition we set $V(s) = 0$ if $s$ is a terminal
# state. We also use a target network to compute $V(s_{t+1})$ for
# added stability. The target network has its weights kept frozen most of
# the time, but is updated with the policy network's weights every so often.
# This is usually a set number of steps but we shall use episodes for
# simplicity.
# 
# 
# 
def batch_learning(memory, policy_net, target_net, optimizer):
    for i in range(Env.LEARNING_BATCH):
        optimize_model(memory, policy_net, target_net, optimizer, i)

    target_net.load_state_dict(policy_net.state_dict())


def optimize_model(memory, policy_net, target_net, optimizer, id=None):
    if len(memory) < Env.BATCH_SIZE:
        return
    transitions = memory.sample(Env.BATCH_SIZE)

    batch = Transition(*zip(*transitions))

    loss = 0.0

    for i in range(len(batch.state)):
        state_action_values = policy_net(batch.state[i][0], batch.state[i][1], batch.state[i][2], batch.state[i][3], batch.state[i][4], batch.state[i][5])
        state_action_value = state_action_values.gather(1, batch.action[i])

        next_state_value = target_net(batch.next_state[i][0], batch.next_state[i][1], batch.next_state[i][2], batch.next_state[i][3], batch.next_state[i][4], batch.next_state[i][5]).max(1)[0].detach()

        expected_state_action_value = (next_state_value * Env.GAMMA) + batch.reward[i]

        loss += F.smooth_l1_loss(state_action_value.view(-1), expected_state_action_value)


    # Optimize the model

    loss /= len(batch.state)
    policy_net.losses.append(loss)
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
