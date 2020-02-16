# -*- coding: utf-8 -*-
import torch
class Env:


    #dqn
    FLOW_SIZE = 60
    EXT_FEATURE_SIZE = 12
    TRAIN_FEATURE_SIZE = 10

    
    RNN_HIDDEN = 8
    GAT_HIDDEN = 32
    GAT_OUTPUT = 8
    EXT_FEATURE_EMBED_HIDDEN = 32
    TRAIN_FEATURE_EMBED_HIDDEN = 32
    FEATURE_EMBED_OUTPUT = 8
    FUSION_HIDDEN = 64
    ACTION_NUM = 5

    DROPOUT_RATE = 0.6

    
    BATCH_SIZE = 128
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    TARGET_UPDATE = 320
    MEMORY_SIZE = 2000
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 100
    LEARNING_GAP = 32

    LEARNING_BATCH = 32
    LEARNING_GAP_TIME = 5400


    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
