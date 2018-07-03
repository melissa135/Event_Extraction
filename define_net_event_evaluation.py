import os
import cPickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Event_Evaluation(nn.Module):
    def __init__(self):
        super(Event_Evaluation, self).__init__()
        self.role_types = len(relation_index)+1
        self.role_size = 8
        encoding_size = 2*128 #
        entity_embedding_size = 16
        event_embedding_size = 16
        self.hidden_size = 128
        self.num_hidden = 1
        tmp_size = 64
    
        self.role_embedding = nn.Embedding(self.role_types, self.role_size)
        self.lstm = nn.LSTM(encoding_size+entity_embedding_size+event_embedding_size+self.role_size,\
                            self.hidden_size, self.num_hidden, bidirectional=True, dropout=0)

        self.linear1 = nn.Linear(2*self.hidden_size,tmp_size)
        self.linear2 = nn.Linear(tmp_size,2)
        self.softmax = nn.LogSoftmax(dim=1)

        self.linearm_1 = nn.Linear(2*self.hidden_size,tmp_size)
        self.linearm_2 = nn.Linear(tmp_size,3)
        self.softmaxm = nn.LogSoftmax(dim=1)

    def forward(self,bilstm_output,event_emb,entity_emb,s_r,range_list,hidden):
        
        none_variable = torch.LongTensor(1,1).zero_()
        none_variable[0][0] = relation_index['NONE']
        none_variable = Variable(none_variable).cuda()
        role_vectors = [ self.role_embedding(none_variable) for i in range(0,bilstm_output.size()[0]) ]

        tigger_variable = torch.LongTensor(1,1).zero_()
        tigger_variable[0][0] = self.role_types - 1
        tigger_variable = Variable(tigger_variable).cuda()
        for i in range(s_r[0],s_r[1]+1) :
            role_vectors[i] = self.role_embedding(tigger_variable)
        
        for rang in range_list :
            rlt_type,[dst_begin,dst_end] = rang
            rlt_variable = torch.LongTensor(1,1).zero_()
            rlt_variable[0][0] = relation_index[rlt_type]
            rlt_variable = Variable(rlt_variable).cuda()
            for i in range(dst_begin,dst_end+1) :
                role_vectors[i] = self.role_embedding(rlt_variable) # N(=1)*L(=1)*role_size

        role_emb = torch.cat(role_vectors,0) # L*N*role_size
        composite_emd = torch.cat((bilstm_output,event_emb,entity_emb,role_emb),2)
        _, (hidden,cell_state) = self.lstm(composite_emd, hidden) #(num_layers * num_directions, batch, hidden_size)
        hidden = hidden.view(1,-1)
        
        x = F.leaky_relu(self.linear1(hidden))
        x = self.linear2(x)
        x = self.softmax(x)

        m = F.leaky_relu(self.linearm_1(hidden))
        m = self.linearm_2(m)
        m = self.softmaxm(m)
            
        return x,m

    def initHidden(self,num_layers=2,batch=1):
        return (Variable(torch.zeros(num_layers*self.num_hidden, batch, self.hidden_size)).cuda(),
                Variable(torch.zeros(num_layers*self.num_hidden, batch, self.hidden_size)).cuda())
    
path_ = os.path.abspath('.')
f = file(path_+'/relation_index', 'r')
relation_index = cPickle.load(f)
relation_index_r = dict()
for key in relation_index.keys() :
    value = relation_index[key]
    relation_index_r[value] = key
