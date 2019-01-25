import os
import cPickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Char_CNN_pretrain(nn.Module):
    def __init__(self, char_dim, event_dim):
        super(Char_CNN_pretrain, self).__init__()
        embedding_size = 16
        kernel = 5
        stride = 1
        padding = 2
        self.conv_out_channel = 64

        self.embedding = nn.Embedding(char_dim, embedding_size)
        self.conv = nn.Conv1d(embedding_size, self.conv_out_channel, kernel, stride, padding)
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.linear = nn.Linear(self.conv_out_channel, event_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embedding(x) # N*L -> N*L*embedding_dim
        x = x.permute(0,2,1) # N*L*embedding_dim -> N*embedding_dim*L
        x = self.conv(x) # N*embedding_dim*L -> N*conv_out_channel*L
        x = self.pooling(x) # N*conv_out_channel*L -> N*conv_out_channel*1
        x = x.view(-1,self.conv_out_channel) # N*conv_out_channel*1 -> N*conv_out_channel
        x = self.linear(x) # N*conv_out_channel -> N*event_dim
        x = self.softmax(x) # N * ( probility of each event )
        return x
