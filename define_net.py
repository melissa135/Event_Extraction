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


class Char_CNN_encode(nn.Module):
    def __init__(self, char_dim):
        super(Char_CNN_encode, self).__init__()
        embedding_size = 16
        kernel = 5
        stride = 1
        padding = 2
        self.conv_out_channel = 64
        
        self.embedding = nn.Embedding(char_dim, embedding_size)
        self.conv = nn.Conv1d(embedding_size, self.conv_out_channel, kernel, stride, padding)
        self.pooling = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = self.embedding(x) # N*L -> N*L*embedding_dim
        x = x.permute(0,2,1) # N*L*embedding_dim -> N*embedding_dim*L
        x = self.conv(x) # N*embedding_dim*L -> N*conv_out_channel*L
        x = self.pooling(x) # N*conv_out_channel*L -> N*conv_out_channel*1
        x = x.view(1,-1,self.conv_out_channel) # N*conv_out_channel*1 -> 1*N*conv_out_channel
        return x
    

class BiLSTM(nn.Module):
    def __init__(self, word_dim, entity_dim):
        super(BiLSTM, self).__init__()
        embedding_size = 192
        entity_embedding_size = 16
        conv_out_channel = 64 # must be same as the one in Char_CNN_encode
        self.hidden_size = 128
        self.num_hidden = 1
        
        self.word_embedding = nn.Embedding(word_dim, embedding_size)
        self.entity_embedding = nn.Embedding(entity_dim, entity_embedding_size)
        self.lstm = nn.LSTM(embedding_size+entity_embedding_size+conv_out_channel, self.hidden_size, self.num_hidden, bidirectional=True, dropout=0)
        self.dropout = nn.Dropout(p=0.0)

    def forward(self, inputs, hidden):
        words,entitys,char_features = inputs
        length = words.size()[1]
        word_emb = self.word_embedding(words) # N*L -> N*L*embedding_size
        word_emb = self.dropout(word_emb)
        word_emb = word_emb.permute(1,0,2) # N*L*embedding_size -> L*N*embedding_size
        entity_emb = self.entity_embedding(entitys) # N*L -> N*L*entity_embedding_size
        entity_emb = self.dropout(entity_emb)
        entity_emb = entity_emb.permute(1,0,2) # N*L*entity_embedding_size -> L*N*entity_embedding_size

        input_ = torch.cat((word_emb, entity_emb, char_features), 2) # L*N*(embedding_size+conv_out_channel)
        # the LSTM can only accept mini-batch
        output, hidden = self.lstm(input_, hidden) # L*N* 2 self.hidden_size
        #output = F.relu(output) # need activation?
        return output, hidden, entity_emb

    def initHidden(self,num_layers=2,batch=1):
        return (Variable(torch.zeros(num_layers*self.num_hidden, batch, self.hidden_size)).cuda(),
                Variable(torch.zeros(num_layers*self.num_hidden, batch, self.hidden_size)).cuda())


class NER_simple(nn.Module):
    def __init__(self, event_dim):
        super(NER_simple, self).__init__()
        self.linear = nn.Linear(256*2,event_dim)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, inputs):
        inputs = inputs.view(-1,256*2)
        inputs = self.linear(inputs)
        inputs = self.softmax(inputs)
        return inputs


class NER_old(nn.Module):
    def __init__(self, event_dim):
        super(NER_old, self).__init__()
        self.event_dim = event_dim
        self.BiLSTM_hidden_size = 2*256 # must be same as previous
        self.NER_hidden_size = event_dim
        self.num_hidden = 1
        self.event_dim = event_dim
        
        self.lstm = nn.LSTMCell(self.BiLSTM_hidden_size+self.event_dim,\
                                self.NER_hidden_size, self.num_hidden)
        self.linear = nn.Linear(self.NER_hidden_size, self.event_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, bilstm_output, ner_softmax, ner_hidden):
        
        ner_output = []
        hidden,cell_state = ner_hidden
        
	for i in range(0,bilstm_output.size()[0]) :
            inputs = torch.cat((bilstm_output[i], ner_softmax), 1) # N*(BiLSTM_hidden+linear_result)
            hidden,cell_state = self.lstm(inputs,(hidden,cell_state)) # two N*self.NER_hidden_size
            hidden = self.linear(hidden) # N*self.NER_hidden_size -> N*event_dim
            ner_softmax = self.softmax(hidden) # N*event_dim
            tmp = ner_softmax.view(-1,self.event_dim)
            ner_output.append(tmp)
            
        ner_output = torch.cat(ner_output,0) # L*event_dim
        return ner_output

    def initHidden(self,batch=1):
        return (Variable(torch.zeros(batch, self.NER_hidden_size)),
                Variable(torch.zeros(batch, self.NER_hidden_size)))

    def initSoftmax(self,batch=1):
        return Variable(torch.zeros(batch, self.event_dim))


class Trigger_Recognition(nn.Module):
    def __init__(self, event_dim):
        super(Trigger_Recognition, self).__init__()
        self.event_dim = event_dim
        self.BiLSTM_hidden_size = 2*128 # must be same as previous
        self.event_dim = event_dim
        self.embedding_size = 16
        tmp_dim = 128

        self.event_embedding = nn.Embedding(event_dim, self.embedding_size)
        #self.layer_norm = nn.LayerNorm(self.BiLSTM_hidden_size+self.embedding_size)
        self.linear = nn.Linear(self.BiLSTM_hidden_size+self.embedding_size, tmp_dim)
        self.linear2 = nn.Linear(tmp_dim, self.event_dim)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, bilstm_output, event_index=0):
        
        ner_output = []
        ner_emb = []
        ner_index = []
        
        event_index = torch.LongTensor([[event_index]]) # N=1 and L=1
        event_emb = self.event_embedding(Variable(event_index).cuda()) # N*L*embedding_size
        event_emb = event_emb.view(-1,self.embedding_size) # N*embedding_size
            
	for i in range(0,bilstm_output.size()[0]) :

            inputs = torch.cat((bilstm_output[i], event_emb), 1) # N*(BiLSTM_hidden+linear_result)
            #inputs = self.layer_norm(inputs)
            inputs = self.dropout(inputs)
            hidden = F.leaky_relu(self.linear(inputs)) # N*hidden_size
            hidden = self.linear2(hidden)
            ner_softmax = self.softmax(hidden) # N*event_dim
            
            row = list(ner_softmax[0])
            for k in range(0,len(row)) :
                row[k] = float(row[k].data)
            row[0] = row[0]*1.5
            event_index = row.index(max(row))
            event_index = torch.LongTensor([[event_index]])
            event_emb = self.event_embedding(Variable(event_index).cuda()) # N*L*embedding_size
            event_emb = event_emb.view(-1,self.embedding_size) # N*embedding_size

            ner_index.append(event_index)
            ner_output.append(ner_softmax)
            ner_emb.append(event_emb.view(-1,1,self.embedding_size))

        ner_index = torch.cat(ner_index,0) # L * 1
        ner_index = ner_index.view(-1) # L
        ner_output = torch.cat(ner_output,0) # L*event_dim
        ner_emb = torch.cat(ner_emb,0) # L*N*embedding_size
        return ner_output,ner_emb,ner_index

    def get_event_embedding(self, event_index) :

        event_index = torch.LongTensor([[event_index]])
        event_emb = self.event_embedding(Variable(event_index).cuda())
        return event_emb


class Relation_ClassificationC(nn.Module):
    def __init__(self, relation_dim):
        super(Relation_ClassificationC, self).__init__()
        self.BiLSTM_hidden_size = 2*128 # must be same as previous
        self.relation_dim = relation_dim
        self.ner_embedding = 16
        tmp_dim = 128
        self.context_relation_dim = 64
        
        self.pooling_src = nn.AdaptiveMaxPool1d(1)
        self.pooling_middle = nn.AdaptiveMaxPool1d(1)
        self.pooling_dst = nn.AdaptiveMaxPool1d(1)
        self.linear = nn.Linear(2*self.ner_embedding+3*self.BiLSTM_hidden_size+5+128, tmp_dim)
        self.linear2 = nn.Linear(tmp_dim, self.relation_dim)

        self.conv_out_dim = 128
        self.conv_3 = nn.Conv1d(self.BiLSTM_hidden_size,self.conv_out_dim,3,1,1)
        self.conv_3r = nn.Conv1d(self.BiLSTM_hidden_size,self.conv_out_dim,3,1,1)

        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(p=0.2)

        self.position_dim = 5
        self.max_position = 16
        self.position_embedding = nn.Embedding(2*self.max_position+1, self.position_dim)
        self.empty_embedding = nn.Embedding(1, self.BiLSTM_hidden_size)
        #self.context_relation_embedding = nn.Embedding(self.relation_dim, self.context_relation_dim)
       
    def forward(self, src_event,src,middle,dst,dst_event,reverse_flag,middle_flag):
        tensor = torch.LongTensor(1,1).zero_()
        tensor[0][0] = middle.size()[0]-1
        if tensor[0][0] > self.max_position :
            tensor[0][0] = self.max_position
        if reverse_flag :
            tensor[0][0] = - tensor[0][0]
        tensor[0][0] = tensor[0][0] + self.max_position
        
        src_event = src_event.permute(1,2,0)
        src_event = self.pooling_src(src_event)
        src = src.permute(1,2,0) # L*N*BiLSTM_hidden_size -> N*BiLSTM_hidden_size*L
        src = self.pooling_src(src) # N*BiLSTM_hidden_size*L -> N*BiLSTM_hidden_size * 1
        if middle_flag :
            middle = self.empty_embedding(Variable(torch.LongTensor([[0]])).cuda())
        middle = middle.permute(1,2,0)
        middle_1 = self.pooling_middle(middle)
        #middle = middle.view(1,-1,1) # N*BiLSTM_hidden*3 -> N* 3 BiLSTM_hidden * 1
        dst = dst.permute(1,2,0)
        dst = self.pooling_dst(dst)
        dst_event = dst_event.permute(1,2,0)
        dst_event = self.pooling_dst(dst_event)

        pe = self.position_embedding(Variable(tensor).cuda())
        pe = pe.view(1,-1,1)

        if not reverse_flag :
            middle_2 = self.conv_3(middle)
        else :
            middle_2 = self.conv_3r(middle)
        middle_2 = self.pooling_middle(middle_2)
        
        x = torch.cat((src_event,src,middle_1,middle_2,dst,dst_event,pe),1) # N * 3 BiLSTM_hidden_size * 1
        x = x.view(1,-1) # N(=1) * 3 BiLSTM_hidden_size
        #x = self.layer_norm(x)
        x = self.dropout(x)
        x = F.leaky_relu(self.linear(x)) # N*relation_dim
        x = self.linear2(x)
        x = self.softmax(x)# N * ( probility of each relation )

        return x


class RC_cnn(nn.Module):
    def __init__(self, relation_dim):
        super(RC_cnn, self).__init__()
        self.BiLSTM_hidden_size = 2*256 # must be same as previous
        self.relation_dim = relation_dim
        self.ner_embedding = 16
        self.conv_out_dim = 128

        self.max_position = 32
        self.position_dim = 5

        self.position_embedding_src = nn.Embedding(2*self.max_position+1, self.position_dim)
        self.position_embedding_dst = nn.Embedding(2*self.max_position+1, self.position_dim)
        self.conv_3 = nn.Conv1d(self.BiLSTM_hidden_size+self.ner_embedding+2*self.position_dim,\
                              self.conv_out_dim,3,1,1)
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.linear = nn.Linear(3*self.conv_out_dim, 128)
        self.linear2 = nn.Linear(128, self.relation_dim)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, bilstm_output,ner_emb,src_range,dst_range):
        
        length = bilstm_output.size()[0]  # L
        tensor_a = torch.LongTensor(1,length).zero_() # N*L
        tensor_b = torch.LongTensor(1,length).zero_() # N*L
        
        for i in range(0,src_range[0]):
            if src_range[0]-i < self.max_position :
                tensor_a[0][i] = i - src_range[0]
            else :
                tensor_a[0][i] = -self.max_position
                
        for i in range(src_range[1]+1,length):
            if i-src_range[1] < self.max_position :
                tensor_a[0][i] = i - src_range[1]
            else :
                tensor_a[0][i] = self.max_position
                
        for i in range(0,dst_range[0]):
            if dst_range[0]-i < self.max_position :
                tensor_b[0][i] = i - dst_range[0]
            else :
                tensor_b[0][i] = -self.max_position
                
        for i in range(dst_range[1]+1,length):
            if i-dst_range[1] < self.max_position :
                tensor_b[0][i] = i - dst_range[1]
            else :
                tensor_b[0][i] = self.max_position

        for i in range(0,length):
            tensor_a[0][i] = tensor_a[0][i] + self.max_position
            tensor_b[0][i] = tensor_b[0][i] + self.max_position
        
        position_index_a = Variable(tensor_a)
        position_index_b = Variable(tensor_b)
        position_output_a = self.position_embedding_src(position_index_a) # N*L*position_dim
        position_output_b = self.position_embedding_dst(position_index_b) # N*L*position_dim
        position_output = torch.cat((position_output_a,position_output_b),2) # N*L*(2*position_dim)
        position_output = position_output.permute(1,0,2) # L*N*(2*position_dim)
        
        input_ = torch.cat((bilstm_output,ner_emb,position_output),2) # L*N*(BiLSTM_hidden_size+ner_embedding+2*position_dim)
        input_ = input_.permute(1,2,0) # N*(BiLSTM_hidden_size+ner_embedding+2*position_dim)*L

        if src_range[0] < dst_range[0] :
            split1,split2 = src_range,dst_range
        else :
            split1,split2 = dst_range,src_range

        input_a = input_[:,:,0:split1[1]+1]
        input_b = input_[:,:,split1[0]:split2[1]+1]
        input_c = input_[:,:,split2[0]:]
        
        input_a = self.conv_3(input_a) # N * conv_out_dim * L
        input_b = self.conv_3(input_b) # N * conv_out_dim * L
        input_c = self.conv_3(input_c) # N * conv_out_dim * L
        
        input_a = self.pooling(input_a) # N * conv_out_dim * 1
        input_b = self.pooling(input_b) # N * conv_out_dim * 1
        input_c = self.pooling(input_c) # N * conv_out_dim * 1

        input_a = input_a.view(1,-1) # N * conv_out_dim
        input_b = input_b.view(1,-1) # N * conv_out_dim
        input_c = input_c.view(1,-1) # N * conv_out_dim
        
        x = torch.cat((input_a,input_b,input_c),1) # N*(3*conv_out_dim)
        x = self.dropout(x)
        x = F.relu(self.linear(x)) # N*relation_dim
        x = self.linear2(x)
        x = self.softmax(x)# N * ( probility of each relation )
        return x


class RC_lstm(nn.Module):
    def __init__(self, relation_dim, marker_type=3):
        super(RC_lstm, self).__init__()
        self.BiLSTM_hidden_size = 2*256 # must be same as previous
        self.relation_dim = relation_dim
        self.ner_embedding = 16
        self.marker_dim = 5
        self.hidden_size = 192
        
        self.marker_embedding = nn.Embedding(marker_type, self.marker_dim)
        self.lstm = nn.LSTM(self.BiLSTM_hidden_size+self.ner_embedding+self.marker_dim, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, relation_dim)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, bilstm_output,ner_emb,src_range,dst_range,hidden_):
        
        length = bilstm_output.size()[0]  # L
        tensor = torch.LongTensor(1,length).zero_() # N*L
        for i in range(src_range[0],src_range[1]+1):
            tensor[0][i] = 1
        for i in range(dst_range[0],dst_range[1]+1):
            tensor[0][i] = 2
        
        marker_index = Variable(tensor)
        marker_output = self.marker_embedding(marker_index) # N*L*marker_dim
        marker_output = marker_output.permute(1,0,2) # L*N*marker_dim
        
        input_ = torch.cat((bilstm_output,ner_emb,marker_output),2) # L*N*(BiLSTM_hidden_size+ner_embedding+marker_dim)
        output, (hidden,c_state) = self.lstm(input_, hidden_) # L*N*self.hidden_size
        output = hidden.view(1,-1) # N*self.hidden_size
        #output = [[1 for i in range(0,self.hidden_size)]]
        #output = Variable(torch.FloatTensor(output))
        output = self.linear(output)
        output = self.softmax(output) # N*relation_dim
        return output, hidden

    def initHidden(self,num_layers=1,batch=1):
        return (Variable(torch.zeros(num_layers, batch, self.hidden_size)),
                Variable(torch.zeros(num_layers, batch, self.hidden_size)))
