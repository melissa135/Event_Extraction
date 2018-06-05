import os
import sys
import torch
import gensim
import cPickle
from random import uniform
import numpy as np
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable
from sentence_set import Sentence_Set
from dataloader_modified import DataLoader
from define_net import Char_CNN_encode,BiLSTM,NER,RC
from define_net_event_assert import *
from entity_event_dict import *


def load_pretrain_vector(word_index,word_embedding):

    filename = os.path.abspath('.') + '/myword2vec'
    my_word2vec = gensim.models.Word2Vec.load(filename)
    not_found = 0
    
    for key in word_index.keys() :
        try :
            #pretrain_vector = []
            pretrain_vector = my_word2vec.wv[key].tolist()
            index = word_index[key]
            #for i in range(0,len(pretrain_vector)):
            #    pretrain_vector[i] = word_embedding[index][i]*0.99 + pretrain_vector[i]*0.01
            #pretrain_vector = word_embedding[index]
            word_embedding[index] = pretrain_vector[:]
        except Exception as e :
            #print e
            not_found = not_found + 1

    print 'There are %d words not found in Word2Vec.'%not_found
    return word_embedding


def has_same_set(seta,setb_all):

    for item in seta.copy() : # use copy() to prevent Set changed size during iteration
        if item[1] == 'None' :
            seta.remove(item)
    
    subsetb = dict()
    for item in setb_all :
        if subsetb.has_key(item[0]) :
            subsetb[item[0]].add((item[1],item[2]))
        else :
            subsetb[item[0]] = set()
            subsetb[item[0]].add((item[1],item[2]))
        
    for value in subsetb.values() :
        if is_same_set(seta,value) :
            return True
    
    return False

def is_same_set(seta,setb):
    
    for item in seta :
        if not item in setb :
            return False
    for item in setb :
        if not item in seta :
            return False
    return True


if __name__ == '__main__':
	
    path_ = os.path.abspath('.')

    number = int(sys.argv[1])

    trainset = Sentence_Set(path_+'/table_train/',new_dict=False)
    testset = Sentence_Set(path_+'/table_test/',new_dict=False)
    char_dim = trainset.get_char_dim()
    word_dim = trainset.get_word_dim()
    entity_dim = trainset.get_entity_dim()
    event_dim = trainset.get_event_dim()
    relation_dim = trainset.get_relation_dim()
    print 'Total %d samples.' % trainset.__len__()
    print 'Char dimension : ' , char_dim
    print 'Word dimension : ' , word_dim
    print 'Entity dimension : ' , entity_dim
    print 'Event dimension : ' , event_dim
    print 'Relation dimension : ' , relation_dim
	
    # the length of samples here are different, so we can't directly use DataLoader provided by PyTorch 
    trainloader = DataLoader(trainset,batch_size=8,shuffle=True)
    '''
    f = file(path_+'/word_index_all', 'r')
    word_index = cPickle.load(f)
    '''
    char_cnn = Char_CNN_encode(char_dim)
    bilstm = BiLSTM(word_dim,entity_dim)
    ner = NER(event_dim)
    rc = RC(relation_dim)
    
    init.xavier_uniform(char_cnn.embedding.weight)
    init.xavier_uniform(char_cnn.conv.weight)
    init.xavier_uniform(bilstm.embedding.weight)
    init.xavier_uniform(bilstm.entity_embedding.weight)
    #init.xavier_uniform(bilstm.lstm.weight)
    init.xavier_uniform(ner.event_embedding.weight)
    init.xavier_uniform(ner.linear.weight)
    init.xavier_uniform(ner.linear2.weight)
    init.xavier_uniform(rc.position_embedding.weight)
    init.xavier_uniform(rc.linear.weight)
    init.xavier_uniform(rc.linear2.weight)
    
    print char_cnn
    print bilstm
    print ner
    print rc

    smp = Simple_Max_Pooling()
    none_embedding = Simple_None_Embedding()
    event_assert = Event_Assert(41,12,59)

    f = file(path_+'/event_index', 'r')
    event_index = cPickle.load(f)
    event_index_r = dict()
    for key in event_index.keys() :
        value = event_index[key]
        event_index_r[value] = key

    f = file(path_+'/entity_index', 'r')
    entity_index = cPickle.load(f)
    entity_index_r = dict()
    for key in entity_index.keys() :
        value = entity_index[key]
        entity_index_r[value] = key
                    
    f = file(path_+'/relation_index', 'r')
    relation_index = cPickle.load(f)
    relation_index_r = dict()
    for key in relation_index.keys() :
        value = relation_index[key]
        relation_index_r[value] = key
                    
    '''
    word_embedding = bilstm.state_dict()['embedding.weight'].numpy()
    word_embedding = load_pretrain_vector(word_index,word_embedding)
    pretrained_weight = np.array(word_embedding)
    bilstm.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))

    for p in bilstm.embedding.parameters():
	p.requires_grad = False
    '''
    #testset = Sentence_Set(path_+'/table_test/',new_dict=False)

    class_weight_ner = [5 for i in range(0,event_dim)] #
    class_weight_ner[0] = 1
    class_weight_ner = torch.FloatTensor(class_weight_ner)
    criterion_ner = nn.NLLLoss(weight=class_weight_ner)
    
    class_weight_rc = [5 for i in range(0,relation_dim)] #
    class_weight_rc[2] = 1
    class_weight_rc = torch.FloatTensor(class_weight_rc)
    criterion_rc = nn.NLLLoss(weight=class_weight_rc)

    class_weight_ea = [1,1] #
    class_weight_ea = torch.FloatTensor(class_weight_ea)
    criterion_ea = nn.MultiMarginLoss() #weight=class_weight_ea
    '''
    criterion_ea_dict = dict()
    
    class_weight_ea = [10,1] #
    class_weight_ea = torch.FloatTensor(class_weight_ea)
    criterion_ea_dict['Development'] = nn.NLLLoss(weight=class_weight_ea)
    class_weight_ea = [1,5] #
    class_weight_ea = torch.FloatTensor(class_weight_ea)
    criterion_ea_dict['Blood_vessel_development'] = nn.NLLLoss(weight=class_weight_ea)
    class_weight_ea = [10,1] #
    class_weight_ea = torch.FloatTensor(class_weight_ea)
    criterion_ea_dict['Growth'] = nn.NLLLoss(weight=class_weight_ea)
    class_weight_ea = [10,1] #
    class_weight_ea = torch.FloatTensor(class_weight_ea)
    criterion_ea_dict['Death'] = nn.NLLLoss(weight=class_weight_ea)
    class_weight_ea = [2,8] #
    class_weight_ea = torch.FloatTensor(class_weight_ea)
    criterion_ea_dict['Cell_death'] = nn.NLLLoss(weight=class_weight_ea)
    class_weight_ea = [25,5] #
    class_weight_ea = torch.FloatTensor(class_weight_ea)
    criterion_ea_dict['Breakdown'] = nn.NLLLoss(weight=class_weight_ea)
    class_weight_ea = [10,2] #
    class_weight_ea = torch.FloatTensor(class_weight_ea)
    criterion_ea_dict['Cell_proliferation'] = nn.NLLLoss(weight=class_weight_ea)
    class_weight_ea = [50,5] #
    class_weight_ea = torch.FloatTensor(class_weight_ea)
    criterion_ea_dict['Cell_division'] = nn.NLLLoss(weight=class_weight_ea)
    class_weight_ea = [30,5] #
    class_weight_ea = torch.FloatTensor(class_weight_ea)
    criterion_ea_dict['Cell_differentiation'] = nn.NLLLoss(weight=class_weight_ea)
    class_weight_ea = [30,5] #
    class_weight_ea = torch.FloatTensor(class_weight_ea)
    criterion_ea_dict['Remodeling'] = nn.NLLLoss(weight=class_weight_ea)
    class_weight_ea = [50,50] #
    class_weight_ea = torch.FloatTensor(class_weight_ea)
    criterion_ea_dict['Reproduction'] = nn.NLLLoss(weight=class_weight_ea)
    class_weight_ea = [10,10] #
    class_weight_ea = torch.FloatTensor(class_weight_ea)
    criterion_ea_dict['Mutation'] = nn.NLLLoss(weight=class_weight_ea)
    class_weight_ea = [4,8] #
    class_weight_ea = torch.FloatTensor(class_weight_ea)
    criterion_ea_dict['Carcinogenesis'] = nn.NLLLoss(weight=class_weight_ea)
    class_weight_ea = [20,2] #
    class_weight_ea = torch.FloatTensor(class_weight_ea)
    criterion_ea_dict['Cell_transformation'] = nn.NLLLoss(weight=class_weight_ea)
    class_weight_ea = [40,40] #
    class_weight_ea = torch.FloatTensor(class_weight_ea)
    criterion_ea_dict['Infection'] = nn.NLLLoss(weight=class_weight_ea)
    class_weight_ea = [30,10] #
    class_weight_ea = torch.FloatTensor(class_weight_ea)
    criterion_ea_dict['Metabolism'] = nn.NLLLoss(weight=class_weight_ea)
    class_weight_ea = [40,10] #
    class_weight_ea = torch.FloatTensor(class_weight_ea)
    criterion_ea_dict['Synthesis'] = nn.NLLLoss(weight=class_weight_ea)
    class_weight_ea = [40,4] #
    class_weight_ea = torch.FloatTensor(class_weight_ea)
    criterion_ea_dict['Catabolism'] = nn.NLLLoss(weight=class_weight_ea)
    class_weight_ea = [50,50] #
    class_weight_ea = torch.FloatTensor(class_weight_ea)
    criterion_ea_dict['Amino_acid_catabolism'] = nn.NLLLoss(weight=class_weight_ea)
    class_weight_ea = [3,30] #
    class_weight_ea = torch.FloatTensor(class_weight_ea)
    criterion_ea_dict['Glycolysis'] = nn.NLLLoss(weight=class_weight_ea)
    class_weight_ea = [4,2] #
    class_weight_ea = torch.FloatTensor(class_weight_ea)
    criterion_ea_dict['Gene_expression'] = nn.NLLLoss(weight=class_weight_ea)
    class_weight_ea = [20,10] #
    class_weight_ea = torch.FloatTensor(class_weight_ea)
    criterion_ea_dict['Transcription'] = nn.NLLLoss(weight=class_weight_ea)
    class_weight_ea = [40,40] #
    class_weight_ea = torch.FloatTensor(class_weight_ea)
    criterion_ea_dict['Translation'] = nn.NLLLoss(weight=class_weight_ea)
    class_weight_ea = [40,4] #
    class_weight_ea = torch.FloatTensor(class_weight_ea)
    criterion_ea_dict['Protein_processing'] = nn.NLLLoss(weight=class_weight_ea)
    class_weight_ea = [20,20] #
    class_weight_ea = torch.FloatTensor(class_weight_ea)
    criterion_ea_dict['Phosphorylation'] = nn.NLLLoss(weight=class_weight_ea)
    class_weight_ea = [10,5] #
    class_weight_ea = torch.FloatTensor(class_weight_ea)
    criterion_ea_dict['Pathway'] = nn.NLLLoss(weight=class_weight_ea)
    class_weight_ea = [2,4] #
    class_weight_ea = torch.FloatTensor(class_weight_ea)
    criterion_ea_dict['Binding'] = nn.NLLLoss(weight=class_weight_ea)
    class_weight_ea = [50,50] #
    class_weight_ea = torch.FloatTensor(class_weight_ea)
    criterion_ea_dict['Dissociation'] = nn.NLLLoss(weight=class_weight_ea)
    class_weight_ea = [3,2] #
    class_weight_ea = torch.FloatTensor(class_weight_ea)
    criterion_ea_dict['Localization'] = nn.NLLLoss(weight=class_weight_ea)
    class_weight_ea = [1,2] #
    class_weight_ea = torch.FloatTensor(class_weight_ea)
    criterion_ea_dict['Regulation'] = nn.NLLLoss(weight=class_weight_ea)
    class_weight_ea = [0.5,0.5] #
    class_weight_ea = torch.FloatTensor(class_weight_ea)
    criterion_ea_dict['Positive_regulation'] = nn.NLLLoss(weight=class_weight_ea)
    class_weight_ea = [0.5,1] #
    class_weight_ea = torch.FloatTensor(class_weight_ea)
    criterion_ea_dict['Negative_regulation'] = nn.NLLLoss(weight=class_weight_ea)
    class_weight_ea = [0.5,1.2] #
    class_weight_ea = torch.FloatTensor(class_weight_ea)
    criterion_ea_dict['Planned_process'] = nn.NLLLoss(weight=class_weight_ea)
    class_weight_ea = [50,50] #
    class_weight_ea = torch.FloatTensor(class_weight_ea)
    criterion_ea_dict['Ubiquitination'] = nn.NLLLoss(weight=class_weight_ea)
    class_weight_ea = [12,8] #
    class_weight_ea = torch.FloatTensor(class_weight_ea)
    criterion_ea_dict['Metastasis'] = nn.NLLLoss(weight=class_weight_ea)
    class_weight_ea = [50,50]  #
    class_weight_ea = torch.FloatTensor(class_weight_ea)
    criterion_ea_dict['Dephosphorylation'] = nn.NLLLoss(weight=class_weight_ea)
    class_weight_ea = [50,50]  #
    class_weight_ea = torch.FloatTensor(class_weight_ea)
    criterion_ea_dict['DNA_demethylation'] = nn.NLLLoss(weight=class_weight_ea)
    class_weight_ea = [50,50]  #
    class_weight_ea = torch.FloatTensor(class_weight_ea)
    criterion_ea_dict['Acetylation'] = nn.NLLLoss(weight=class_weight_ea)
    class_weight_ea = [50,50]  #
    class_weight_ea = torch.FloatTensor(class_weight_ea)
    criterion_ea_dict['DNA_methylation'] = nn.NLLLoss(weight=class_weight_ea)
    class_weight_ea = [50,50]  #
    class_weight_ea = torch.FloatTensor(class_weight_ea)
    criterion_ea_dict['Glycosylation'] = nn.NLLLoss(weight=class_weight_ea)
    '''
    optimizer = optim.Adam(list(char_cnn.parameters())+\
                           list(bilstm.parameters())+\
                           list(ner.parameters())+\
                           list(rc.parameters())+\
                           list(event_assert.parameters()), lr=0.007, weight_decay=0.0002)
    
    for epoch in range(50): #
        pos_event_score,neg_event_score = [],[]

        running_loss = 0.0
        loss_a,loss_b,loss_c = 0.0,0.0,0.0
        valid_count, invalid_count = 0.0, 0.0

        freq_dict = { 'Development':[0,0], 'Blood_vessel_development':[0,0], 'Growth':[0,0], 'Death':[0,0],
                      'Cell_death':[0,0], 'Breakdown':[0,0], 'Cell_proliferation':[0,0], 'Cell_division':[0,0], 'Cell_differentiation':[0,0],
                      'Remodeling':[0,0], 'Reproduction':[0,0], 'Mutation':[0,0], 'Carcinogenesis':[0,0], 'Cell_transformation':[0,0],
                      'Metastasis':[0,0], 'Infection':[0,0], 'Metabolism':[0,0], 'Synthesis':[0,0], 'Catabolism':[0,0],
                      'Amino_acid_catabolism':[0,0], 'Glycolysis':[0,0], 'Gene_expression':[0,0], 'Transcription':[0,0], 'Translation':[0,0],
                      'Protein_processing':[0,0], 'Phosphorylation':[0,0], 'Pathway':[0,0], 'Binding':[0,0], 'Dissociation':[0,0],
                      'Localization':[0,0], 'Regulation':[0,0], 'Positive_regulation':[0,0], 'Negative_regulation':[0,0], 'Planned_process':[0,0],
                      'Ubiquitination':[0,0], 'Dephosphorylation':[0,0], 'DNA_demethylation':[0,0], 'Acetylation':[0,0], 'DNA_methylation':[0,0],
                      'Glycosylation':[0,0] }
        
        for i,batch in enumerate(trainloader,0):

            loss = 0
            optimizer.zero_grad()
            
            for data in batch: # due to we have modified the defination of batch, the batch here is a list

                input_word, input_entity, input_char, target, entity_loc, event_loc, relation, event_para = data
                input_word, input_entity, target = Variable(input_word),Variable(input_entity),Variable(target)
                
                char_encode = []
                for chars in input_char :
                    chars = char_cnn(Variable(chars))
                    char_encode.append(chars)
                char_encode = torch.cat(char_encode,0) # L*N*conv_out_channel

                hidden1,hidden2 = bilstm.initHidden()
		bilstm_output,hidden,entity_emb = bilstm((input_word,input_entity,char_encode),(hidden1,hidden2))
		ner_output,ner_emb,ner_index = ner(bilstm_output)
                
		target = target.view(-1) # L of indices
		loss = loss + criterion_ner(ner_output,target) # NLLloss only accept 1-dimension
                loss_a = loss_a + criterion_ner(ner_output,target)
                
                e_loc = dict(entity_loc.items()+event_loc.items())

                event_para_ = dict()
                rc_hidden_dict = dict()
                src_type = dict()
                dst_type = dict()

                for src_key in event_loc.keys() :
                    src_range = event_loc[src_key]
                    src_begin,src_end = src_range
                    if int(target[src_begin]) != int(ner_index[src_begin]) :
                        continue
                    src_name = event_index_r[int(target[src_begin])].split('-')[0]
                    src_type[src_key] = src_name
                    event_para_[src_key] = set()
                
		for rlt in relation.keys() :
                    
                    src_key = rlt[0]
                    dst_key = rlt[1]
                    rlt_target = Variable(relation[rlt])
                    src_range = e_loc[src_key]
                    dst_range = e_loc[dst_key]
                    
                    src_begin,src_end = src_range
                    dst_begin,dst_end = dst_range
                    # if the prediction of event is incorrect, ignore them
                    correct_event = True
                    for j in range(src_begin,src_end+1):
                        if int(target[j]) != int(ner_index[j]) :
                            correct_event = False
                    if event_loc.has_key(dst_key):
                        for j in range(dst_begin,dst_end+1):
                            if int(target[j]) != int(ner_index[j]) :
                                correct_event = False
                    if not correct_event :
                        invalid_count = invalid_count + 1
                        continue

                    valid_count = valid_count + 1
                    
                    src = bilstm_output[src_begin:src_end+1]
                    src_event = ner_emb[src_begin:src_end+1]
                    dst = bilstm_output[dst_begin:dst_end+1]
                    
                    event_flag = False
                    if event_loc.has_key(dst_key):
                        dst_event = ner_emb[dst_begin:dst_end+1]
                        dst_name = event_index_r[ner_index[dst_begin]].split('-')[0]
                        event_flag = True
                    else :
                        dst_event = entity_emb[dst_begin:dst_end+1]
                        dst_name = entity_index_r[input_entity.data[0][dst_begin]].split('-')[0]
                    dst_type[dst_key] = dst_name
                    
                    reverse_flag = False
                    if  src_end+1 < dst_begin:
                        middle = bilstm_output[src_end+1:dst_begin]
                    elif dst_end < src_begin-1:
                        reverse_flag = True
                        middle = bilstm_output[dst_end+1:src_begin]
                    else : # adjacent or overlapped
                        middle = Variable(torch.zeros(1,1,128*2)) # L(=1)*N*2 self.hidden_size
                                     
                    rc_output,rc_hidden = rc(src_event,src,middle,dst,dst_event,reverse_flag)
                    '''
                    #hidden = rc.initHidden()
                    rc_output = rc(bilstm_output,ner_emb,src_range,dst_range)
                    '''                                    
                    loss = loss + criterion_rc(rc_output,rlt_target)/10.0
                    loss_b = loss_b + criterion_rc(rc_output,rlt_target)/10.0

                    row = rc_output.data
                    this_row = list(row[0])
                    index = this_row.index(max(this_row))
                    current_type = relation_index_r[index]
                    current_strength = this_row[index]

                    if relation[(src_key,dst_key)][0] != index :
                        continue
                            
                    if current_type != 'NONE' :
                        dst = smp(dst)
                        rc_hidden_dict[(src_key,dst_key)] = (dst,current_strength)#rc_hidden,current_strength)#rc_output#Variable(row[:])
                        event_para_[src_key].add((current_type,dst_key))
                
                dst_type['None'] = 'None'
                
                pos_target = Variable(torch.LongTensor([1]))
                neg_target = Variable(torch.LongTensor([0]))

                loss_ea_resize = 1.0
                ############################################################
                for src_key in event_para_.keys() :

                    src_ = torch.LongTensor(1,1).zero_()
                    src_[0][0] = event_dict[src_type[src_key]]
                    src_ = Variable(src_)

                    #criterion_ea = criterion_ea_dict[src_type[src_key]]
                    
                    if src_type[src_key] == 'Development' or \
                       src_type[src_key] == 'Growth' or \
                       src_type[src_key] == 'Death' or \
                       src_type[src_key] == 'Breakdown' or \
                       src_type[src_key] == 'Cell_proliferation' or \
                       src_type[src_key] == 'Cell_division' or \
                       src_type[src_key] == 'Remodeling' or \
                       src_type[src_key] == 'Reproduction' or \
                       src_type[src_key] == 'Metabolism' or \
                       src_type[src_key] == 'Synthesis' or \
                       src_type[src_key] == 'Catabolism' or \
                       src_type[src_key] == 'Transcription' or \
                       src_type[src_key] == 'Translation' or \
                       src_type[src_key] == 'Protein_processing':
                        
                        themes = []
                        for para in event_para_[src_key]:
                            if para[0] == 'Theme' :
                                themes.append(para)

                        pos_count,neg_count = 1.0,1.0
                        for theme in themes :
                            rlt_1 = torch.LongTensor(1,1).zero_()
                            rlt_1[0][0] = relation_index['Theme']
                            rlt_1 = Variable(rlt_1)
                            dst_key_1 = theme[1]
                            dst_1 = torch.LongTensor(1,1).zero_()
                            dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                            dst_1 = Variable(dst_1)
                            default_prob = (none_embedding(relation_dict['Theme']),-5.12)
                            if has_same_set({theme},event_para[src_key]) :
                                pos_count = pos_count + 1
                            else :
                                neg_count = neg_count + 1
                        if pos_count < neg_count :
                            min_count = pos_count
                        else :
                            min_count = neg_count
                        pos_count = min_count/pos_count
                        neg_count = min_count/neg_count
                                  
                        for theme in themes :
                            rlt_1 = torch.LongTensor(1,1).zero_()
                            rlt_1[0][0] = relation_index['Theme']
                            rlt_1 = Variable(rlt_1)
                            dst_key_1 = theme[1]
                            dst_1 = torch.LongTensor(1,1).zero_()
                            dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                            dst_1 = Variable(dst_1)
                            default_prob = (none_embedding(relation_dict['Theme']),-5.12)

                            r = event_assert(src_,[(rlt_1,dst_1,rc_hidden_dict.get((src_key,dst_key_1),default_prob))] )
                            if has_same_set({theme},event_para[src_key]) :
                                if uniform(0,1) > pos_count :
                                    continue
                                loss = loss + criterion_ea(r,pos_target)/loss_ea_resize
                                loss_c = loss_c + criterion_ea(r,pos_target)/loss_ea_resize
                                freq_dict[src_type[src_key]][0] = freq_dict[src_type[src_key]][0] + 1
                            else :
                                if uniform(0,1) > neg_count :
                                    continue
                                loss = loss + criterion_ea(r,neg_target)/loss_ea_resize
                                loss_c = loss_c + criterion_ea(r,neg_target)/loss_ea_resize
                                freq_dict[src_type[src_key]][1] = freq_dict[src_type[src_key]][1] + 1

                            if has_same_set({theme},event_para[src_key]) :
                                pos_event_score.append(r.data[0][1])
                            else :
                                neg_event_score.append(r.data[0][1])
                                
                    if src_type[src_key] == 'Cell_death' or \
                       src_type[src_key] == 'Amino_acid_catabolism' or \
                       src_type[src_key] == 'Glycolysis':
                        
                        themes = []
                        for para in event_para_[src_key]:
                            if para[0] == 'Theme' :
                                themes.append(para)

                        pos_count,neg_count = 1.0,1.0
                        for theme in themes + [('Theme','None')] :
                            rlt_1 = torch.LongTensor(1,1).zero_()
                            rlt_1[0][0] = relation_index['Theme']
                            rlt_1 = Variable(rlt_1)
                            dst_key_1 = theme[1]
                            dst_1 = torch.LongTensor(1,1).zero_()
                            dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                            dst_1 = Variable(dst_1)
                            default_prob = (none_embedding(relation_dict['Theme']),-5.12)
                            if has_same_set({theme},event_para[src_key]) :
                                pos_count = pos_count + 1
                            else :
                                neg_count = neg_count + 1
                        if pos_count < neg_count :
                            min_count = pos_count
                        else :
                            min_count = neg_count
                        pos_count = min_count/pos_count
                        neg_count = min_count/neg_count
                                
                        for theme in themes + [('Theme','None')] :
                            rlt_1 = torch.LongTensor(1,1).zero_()
                            rlt_1[0][0] = relation_index['Theme']
                            rlt_1 = Variable(rlt_1)
                            dst_key_1 = theme[1]
                            dst_1 = torch.LongTensor(1,1).zero_()
                            dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                            dst_1 = Variable(dst_1)
                            default_prob = (none_embedding(relation_dict['Theme']),-5.12)
                            
                            r = event_assert(src_,[(rlt_1,dst_1,rc_hidden_dict.get((src_key,dst_key_1),default_prob))] )
                            if has_same_set({theme},event_para[src_key]) :
                                if uniform(0,1) > pos_count :
                                    continue
                                loss = loss + criterion_ea(r,pos_target)/loss_ea_resize
                                loss_c = loss_c + criterion_ea(r,pos_target)/loss_ea_resize
                                freq_dict[src_type[src_key]][0] = freq_dict[src_type[src_key]][0] + 1
                            else :
                                if uniform(0,1) > neg_count :
                                    continue
                                loss = loss + criterion_ea(r,neg_target)/loss_ea_resize
                                loss_c = loss_c + criterion_ea(r,neg_target)/loss_ea_resize
                                freq_dict[src_type[src_key]][1] = freq_dict[src_type[src_key]][1] + 1

                            if has_same_set({theme},event_para[src_key]) :
                                pos_event_score.append(r.data[0][1])
                            else :
                                neg_event_score.append(r.data[0][1])

                    if src_type[src_key] == 'Cell_differentiation' or \
                       src_type[src_key] == 'Cell_transformation':
                        
                        themes = []
                        atlocs = []
                        for para in event_para_[src_key]:
                            if para[0] == 'Theme' :
                                themes.append(para)
                            if para[0] == 'AtLoc' :
                                atlocs.append(para)
                        
                        pos_count,neg_count = 1.0,1.0
                        for theme in themes :
                            rlt_1 = torch.LongTensor(1,1).zero_()
                            rlt_1[0][0] = relation_index['Theme']
                            rlt_1 = Variable(rlt_1)
                            dst_key_1 = theme[1]
                            dst_1 = torch.LongTensor(1,1).zero_()
                            dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                            dst_1 = Variable(dst_1)
                            default_prob1 = (none_embedding(relation_dict['Theme']),-5.12)
                            for atloc in atlocs + [('AtLoc','None')] :
                                rlt_2 = torch.LongTensor(1,1).zero_()
                                rlt_2[0][0] = relation_index['AtLoc']
                                rlt_2 = Variable(rlt_2)
                                dst_key_2 = atloc[1]
                                dst_2 = torch.LongTensor(1,1).zero_()
                                dst_2[0][0] = e_dict[dst_type[dst_key_2]]
                                dst_2 = Variable(dst_2)
                                default_prob2 = (none_embedding(relation_dict['AtLoc']),-5.12)
                                if has_same_set({theme},event_para[src_key]) :
                                    pos_count = pos_count + 1
                                else :
                                    neg_count = neg_count + 1
                        if pos_count < neg_count :
                            min_count = pos_count
                        else :
                            min_count = neg_count
                        pos_count = min_count/pos_count
                        neg_count = min_count/neg_count
                                
                        for theme in themes :
                            rlt_1 = torch.LongTensor(1,1).zero_()
                            rlt_1[0][0] = relation_index['Theme']
                            rlt_1 = Variable(rlt_1)
                            dst_key_1 = theme[1]
                            dst_1 = torch.LongTensor(1,1).zero_()
                            dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                            dst_1 = Variable(dst_1)
                            default_prob1 = (none_embedding(relation_dict['Theme']),-5.12)
                            for atloc in atlocs + [('AtLoc','None')] :
                                rlt_2 = torch.LongTensor(1,1).zero_()
                                rlt_2[0][0] = relation_index['AtLoc']
                                rlt_2 = Variable(rlt_2)
                                dst_key_2 = atloc[1]
                                dst_2 = torch.LongTensor(1,1).zero_()
                                dst_2[0][0] = e_dict[dst_type[dst_key_2]]
                                dst_2 = Variable(dst_2)
                                default_prob2 = (none_embedding(relation_dict['AtLoc']),-5.12)

                                r = event_assert(src_,[(rlt_1,dst_1,rc_hidden_dict.get((src_key,dst_key_1),default_prob1)),\
                                                       (rlt_2,dst_2,rc_hidden_dict.get((src_key,dst_key_2),default_prob2))] )
                                if has_same_set({theme,atloc},event_para[src_key]) :
                                    if uniform(0,1) > pos_count :
                                        continue
                                    loss = loss + criterion_ea(r,pos_target)/loss_ea_resize
                                    loss_c = loss_c + criterion_ea(r,pos_target)/loss_ea_resize
                                    freq_dict[src_type[src_key]][0] = freq_dict[src_type[src_key]][0] + 1
                                else :
                                    if uniform(0,1) > neg_count :
                                        continue
                                    loss = loss + criterion_ea(r,neg_target)/loss_ea_resize
                                    loss_c = loss_c + criterion_ea(r,neg_target)/loss_ea_resize
                                    freq_dict[src_type[src_key]][1] = freq_dict[src_type[src_key]][1] + 1

                                if has_same_set({theme,atloc},event_para[src_key]) :
                                    pos_event_score.append(r.data[0][1])
                                else :
                                    neg_event_score.append(r.data[0][1])

                    if src_type[src_key] == 'Blood_vessel_development' or \
                       src_type[src_key] == 'Carcinogenesis' :
                        
                        themes = []
                        atlocs = []
                        for para in event_para_[src_key]:
                            if para[0] == 'Theme' :
                                themes.append(para)
                            if para[0] == 'AtLoc' :
                                atlocs.append(para)

                        pos_count,neg_count = 1.0,1.0
                        for theme in themes + [('Theme','None')] :
                            rlt_1 = torch.LongTensor(1,1).zero_()
                            rlt_1[0][0] = relation_index['Theme']
                            rlt_1 = Variable(rlt_1)
                            dst_key_1 = theme[1]
                            dst_1 = torch.LongTensor(1,1).zero_()
                            dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                            dst_1 = Variable(dst_1)
                            default_prob1 = (none_embedding(relation_dict['Theme']),-5.12)
                            for atloc in atlocs + [('AtLoc','None')] :
                                rlt_2 = torch.LongTensor(1,1).zero_()
                                rlt_2[0][0] = relation_index['AtLoc']
                                rlt_2 = Variable(rlt_2)
                                dst_key_2 = atloc[1]
                                dst_2 = torch.LongTensor(1,1).zero_()
                                dst_2[0][0] = e_dict[dst_type[dst_key_2]]
                                dst_2 = Variable(dst_2)
                                default_prob2 = (none_embedding(relation_dict['AtLoc']),-5.12)
                                if has_same_set({theme},event_para[src_key]) :
                                    pos_count = pos_count + 1
                                else :
                                    neg_count = neg_count + 1
                        if pos_count < neg_count :
                            min_count = pos_count
                        else :
                            min_count = neg_count
                        pos_count = min_count/pos_count
                        neg_count = min_count/neg_count
                                
                        for theme in themes + [('Theme','None')] :
                            rlt_1 = torch.LongTensor(1,1).zero_()
                            rlt_1[0][0] = relation_index['Theme']
                            rlt_1 = Variable(rlt_1)
                            dst_key_1 = theme[1]
                            dst_1 = torch.LongTensor(1,1).zero_()
                            dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                            dst_1 = Variable(dst_1)
                            default_prob1 = (none_embedding(relation_dict['Theme']),-5.12)
                            for atloc in atlocs + [('AtLoc','None')] :
                                rlt_2 = torch.LongTensor(1,1).zero_()
                                rlt_2[0][0] = relation_index['AtLoc']
                                rlt_2 = Variable(rlt_2)
                                dst_key_2 = atloc[1]
                                dst_2 = torch.LongTensor(1,1).zero_()
                                dst_2[0][0] = e_dict[dst_type[dst_key_2]]
                                dst_2 = Variable(dst_2)
                                default_prob2 = (none_embedding(relation_dict['AtLoc']),-5.12)

                                r = event_assert(src_,[(rlt_1,dst_1,rc_hidden_dict.get((src_key,dst_key_1),default_prob1)),\
                                                       (rlt_2,dst_2,rc_hidden_dict.get((src_key,dst_key_2),default_prob2))] )
                                if has_same_set({theme,atloc},event_para[src_key]) :
                                    if uniform(0,1) > pos_count :
                                        continue
                                    loss = loss + criterion_ea(r,pos_target)/loss_ea_resize
                                    loss_c = loss_c + criterion_ea(r,pos_target)/loss_ea_resize
                                    freq_dict[src_type[src_key]][0] = freq_dict[src_type[src_key]][0] + 1
                                else :
                                    if uniform(0,1) > neg_count :
                                        continue
                                    loss = loss + criterion_ea(r,neg_target)/loss_ea_resize
                                    loss_c = loss_c + criterion_ea(r,neg_target)/loss_ea_resize
                                    freq_dict[src_type[src_key]][1] = freq_dict[src_type[src_key]][1] + 1

                                if has_same_set({theme,atloc},event_para[src_key]) :
                                    pos_event_score.append(r.data[0][1])
                                else :
                                    neg_event_score.append(r.data[0][1])

                    if src_type[src_key] == 'Mutation' :
                        themes = []
                        atlocs = []
                        sites = []
                        for para in event_para_[src_key]:
                            if para[0] == 'Theme' :
                                themes.append(para)
                            if para[0] == 'AtLoc' :
                                atlocs.append(para)
                            if para[0] == 'Site' :
                                sites.append(para)

                        pos_count,neg_count = 1.0,1.0
                        for theme in themes :
                            rlt_1 = torch.LongTensor(1,1).zero_()
                            rlt_1[0][0] = relation_index['Theme']
                            rlt_1 = Variable(rlt_1)
                            dst_key_1 = theme[1]
                            dst_1 = torch.LongTensor(1,1).zero_()
                            dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                            dst_1 = Variable(dst_1)
                            default_prob1 = (none_embedding(relation_dict['Theme']),-5.12)
                            for atloc in atlocs + [('AtLoc','None')] :
                                rlt_2 = torch.LongTensor(1,1).zero_()
                                rlt_2[0][0] = relation_index['AtLoc']
                                rlt_2 = Variable(rlt_2)
                                dst_key_2 = atloc[1]
                                dst_2 = torch.LongTensor(1,1).zero_()
                                dst_2[0][0] = e_dict[dst_type[dst_key_2]]
                                dst_2 = Variable(dst_2)
                                default_prob2 = (none_embedding(relation_dict['AtLoc']),-5.12)
                                for site in sites + [('Site','None')] :
                                    rlt_3 = torch.LongTensor(1,1).zero_()
                                    rlt_3[0][0] = relation_index['Site']
                                    rlt_3 = Variable(rlt_3)
                                    dst_key_3 = site[1]
                                    dst_3 = torch.LongTensor(1,1).zero_()
                                    dst_3[0][0] = e_dict[dst_type[dst_key_3]]
                                    dst_3 = Variable(dst_3)
                                    default_prob3 = (none_embedding(relation_dict['Site']),-5.12)
                                    if has_same_set({theme},event_para[src_key]) :
                                        pos_count = pos_count + 1
                                    else :
                                        neg_count = neg_count + 1
                        if pos_count < neg_count :
                            min_count = pos_count
                        else :
                            min_count = neg_count
                        pos_count = min_count/pos_count
                        neg_count = min_count/neg_count
                                
                        for theme in themes :
                            rlt_1 = torch.LongTensor(1,1).zero_()
                            rlt_1[0][0] = relation_index['Theme']
                            rlt_1 = Variable(rlt_1)
                            dst_key_1 = theme[1]
                            dst_1 = torch.LongTensor(1,1).zero_()
                            dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                            dst_1 = Variable(dst_1)
                            default_prob1 = (none_embedding(relation_dict['Theme']),-5.12)
                            for atloc in atlocs + [('AtLoc','None')] :
                                rlt_2 = torch.LongTensor(1,1).zero_()
                                rlt_2[0][0] = relation_index['AtLoc']
                                rlt_2 = Variable(rlt_2)
                                dst_key_2 = atloc[1]
                                dst_2 = torch.LongTensor(1,1).zero_()
                                dst_2[0][0] = e_dict[dst_type[dst_key_2]]
                                dst_2 = Variable(dst_2)
                                default_prob2 = (none_embedding(relation_dict['AtLoc']),-5.12)
                                for site in sites + [('Site','None')] :
                                    rlt_3 = torch.LongTensor(1,1).zero_()
                                    rlt_3[0][0] = relation_index['Site']
                                    rlt_3 = Variable(rlt_3)
                                    dst_key_3 = site[1]
                                    dst_3 = torch.LongTensor(1,1).zero_()
                                    dst_3[0][0] = e_dict[dst_type[dst_key_3]]
                                    dst_3 = Variable(dst_3)
                                    default_prob3 = (none_embedding(relation_dict['Site']),-5.12)

                                    r = event_assert(src_,[(rlt_1,dst_1,rc_hidden_dict.get((src_key,dst_key_1),default_prob1)),\
                                                           (rlt_2,dst_2,rc_hidden_dict.get((src_key,dst_key_2),default_prob2)),\
                                                           (rlt_3,dst_3,rc_hidden_dict.get((src_key,dst_key_3),default_prob3))] )
                                    if has_same_set({theme,atloc,site},event_para[src_key]) :
                                        if uniform(0,1) > pos_count :
                                            continue
                                        loss = loss + criterion_ea(r,pos_target)/loss_ea_resize
                                        loss_c = loss_c + criterion_ea(r,pos_target)/loss_ea_resize
                                        freq_dict[src_type[src_key]][0] = freq_dict[src_type[src_key]][0] + 1
                                    else :
                                        if uniform(0,1) > neg_count :
                                            continue
                                        loss = loss + criterion_ea(r,neg_target)/loss_ea_resize
                                        loss_c = loss_c + criterion_ea(r,neg_target)/loss_ea_resize
                                        freq_dict[src_type[src_key]][1] = freq_dict[src_type[src_key]][1] + 1

                                    if has_same_set({theme,atloc,site},event_para[src_key]) :
                                        pos_event_score.append(r.data[0][1])
                                    else :
                                        neg_event_score.append(r.data[0][1])

                    if src_type[src_key] == 'Metastasis' :
                        themes = []
                        tolocs = []
                        for para in event_para_[src_key]:
                            if para[0] == 'Theme' :
                                themes.append(para)
                            if para[0] == 'ToLoc' :
                                tolocs.append(para)

                        pos_count,neg_count = 1.0,1.0
                        for theme in themes + [('Theme','None')] :
                            rlt_1 = torch.LongTensor(1,1).zero_()
                            rlt_1[0][0] = relation_index['Theme']
                            rlt_1 = Variable(rlt_1)
                            dst_key_1 = theme[1]
                            dst_1 = torch.LongTensor(1,1).zero_()
                            dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                            dst_1 = Variable(dst_1)
                            default_prob1 = (none_embedding(relation_dict['Theme']),-5.12)
                            for toloc in tolocs :
                                rlt_2 = torch.LongTensor(1,1).zero_()
                                rlt_2[0][0] = relation_index['ToLoc']
                                rlt_2 = Variable(rlt_2)
                                dst_key_2 = toloc[1]
                                dst_2 = torch.LongTensor(1,1).zero_()
                                dst_2[0][0] = e_dict[dst_type[dst_key_2]]
                                dst_2 = Variable(dst_2)
                                default_prob = (none_embedding(relation_dict['ToLoc']),-5.12)
                                if has_same_set({theme},event_para[src_key]) :
                                    pos_count = pos_count + 1
                                else :
                                    neg_count = neg_count + 1
                        if pos_count < neg_count :
                            min_count = pos_count
                        else :
                            min_count = neg_count
                        pos_count = min_count/pos_count
                        neg_count = min_count/neg_count
                                
                        for theme in themes + [('Theme','None')] :
                            rlt_1 = torch.LongTensor(1,1).zero_()
                            rlt_1[0][0] = relation_index['Theme']
                            rlt_1 = Variable(rlt_1)
                            dst_key_1 = theme[1]
                            dst_1 = torch.LongTensor(1,1).zero_()
                            dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                            dst_1 = Variable(dst_1)
                            default_prob1 = (none_embedding(relation_dict['Theme']),-5.12)
                            for toloc in tolocs :
                                rlt_2 = torch.LongTensor(1,1).zero_()
                                rlt_2[0][0] = relation_index['ToLoc']
                                rlt_2 = Variable(rlt_2)
                                dst_key_2 = toloc[1]
                                dst_2 = torch.LongTensor(1,1).zero_()
                                dst_2[0][0] = e_dict[dst_type[dst_key_2]]
                                dst_2 = Variable(dst_2)
                                default_prob = (none_embedding(relation_dict['ToLoc']),-5.12)

                                r = event_assert(src_,[(rlt_1,dst_1,rc_hidden_dict.get((src_key,dst_key_1),default_prob1)),\
                                                       (rlt_2,dst_2,rc_hidden_dict.get((src_key,dst_key_2),default_prob2))] )
                                if has_same_set({theme,toloc},event_para[src_key]) :
                                    if uniform(0,1) > pos_count :
                                        continue
                                    loss = loss + criterion_ea(r,pos_target)/loss_ea_resize
                                    loss_c = loss_c + criterion_ea(r,pos_target)/loss_ea_resize
                                    freq_dict[src_type[src_key]][0] = freq_dict[src_type[src_key]][0] + 1
                                else :
                                    if uniform(0,1) > neg_count :
                                        continue
                                    loss = loss + criterion_ea(r,neg_target)/loss_ea_resize
                                    loss_c = loss_c + criterion_ea(r,neg_target)/loss_ea_resize
                                    freq_dict[src_type[src_key]][1] = freq_dict[src_type[src_key]][1] + 1

                                if has_same_set({theme,toloc},event_para[src_key]) :
                                    pos_event_score.append(r.data[0][1])
                                else :
                                    neg_event_score.append(r.data[0][1])

                    if src_type[src_key] == 'Infection' :
                        themes = []
                        ptps = []
                        for para in event_para_[src_key]:
                            if para[0] == 'Theme' :
                                themes.append(para)
                            if para[0] == 'Participant' :
                                ptps.append(para)

                        pos_count,neg_count = 1.0,1.0
                        for theme in themes + [('Theme','None')] :
                            rlt_1 = torch.LongTensor(1,1).zero_()
                            rlt_1[0][0] = relation_index['Theme']
                            rlt_1 = Variable(rlt_1)
                            dst_key_1 = theme[1]
                            dst_1 = torch.LongTensor(1,1).zero_()
                            dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                            dst_1 = Variable(dst_1)
                            default_prob1 = (none_embedding(relation_dict['Theme']),-5.12)
                            for ptp in ptps + [('Participant','None')] :
                                rlt_2 = torch.LongTensor(1,1).zero_()
                                rlt_2[0][0] = relation_index['Participant']
                                rlt_2 = Variable(rlt_2)
                                dst_key_2 = ptp[1]
                                dst_2 = torch.LongTensor(1,1).zero_()
                                dst_2[0][0] = e_dict[dst_type[dst_key_2]]
                                dst_2 = Variable(dst_2)
                                default_prob2 = (none_embedding(relation_dict['Participant']),-5.12)
                                if has_same_set({theme},event_para[src_key]) :
                                    pos_count = pos_count + 1
                                else :
                                    neg_count = neg_count + 1
                        if pos_count < neg_count :
                            min_count = pos_count
                        else :
                            min_count = neg_count
                        pos_count = min_count/pos_count
                        neg_count = min_count/neg_count
                                
                        for theme in themes + [('Theme','None')] :
                            rlt_1 = torch.LongTensor(1,1).zero_()
                            rlt_1[0][0] = relation_index['Theme']
                            rlt_1 = Variable(rlt_1)
                            dst_key_1 = theme[1]
                            dst_1 = torch.LongTensor(1,1).zero_()
                            dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                            dst_1 = Variable(dst_1)
                            default_prob1 = (none_embedding(relation_dict['Theme']),-5.12)
                            for ptp in ptps + [('Participant','None')] :
                                rlt_2 = torch.LongTensor(1,1).zero_()
                                rlt_2[0][0] = relation_index['Participant']
                                rlt_2 = Variable(rlt_2)
                                dst_key_2 = ptp[1]
                                dst_2 = torch.LongTensor(1,1).zero_()
                                dst_2[0][0] = e_dict[dst_type[dst_key_2]]
                                dst_2 = Variable(dst_2)
                                default_prob2 = (none_embedding(relation_dict['Participant']),-5.12)

                                r = event_assert(src_,[(rlt_1,dst_1,rc_hidden_dict.get((src_key,dst_key_1),default_prob1)),\
                                                       (rlt_2,dst_2,rc_hidden_dict.get((src_key,dst_key_2),default_prob2))] )
                                if has_same_set({theme,ptp},event_para[src_key]) :
                                    if uniform(0,1) > pos_count :
                                        continue
                                    loss = loss + criterion_ea(r,pos_target)/loss_ea_resize
                                    loss_c = loss_c + criterion_ea(r,pos_target)/loss_ea_resize
                                    freq_dict[src_type[src_key]][0] = freq_dict[src_type[src_key]][0] + 1
                                else :
                                    if uniform(0,1) > neg_count :
                                        continue
                                    loss = loss + criterion_ea(r,neg_target)/loss_ea_resize
                                    loss_c = loss_c + criterion_ea(r,neg_target)/loss_ea_resize
                                    freq_dict[src_type[src_key]][1] = freq_dict[src_type[src_key]][1] + 1

                                if has_same_set({theme,ptp},event_para[src_key]) :
                                    pos_event_score.append(r.data[0][1])
                                else :
                                    neg_event_score.append(r.data[0][1])

                    if src_type[src_key] == 'Gene_expression' :
                        themes = []
                        theme2s = []
                        for para in event_para_[src_key]:
                            if para[0] == 'Theme' :
                                themes.append(para)
                            if para[0] == 'Theme2' :
                                theme2s.append(para)

                        pos_count,neg_count = 1.0,1.0
                        for theme in themes :
                            rlt_1 = torch.LongTensor(1,1).zero_()
                            rlt_1[0][0] = relation_index['Theme']
                            rlt_1 = Variable(rlt_1)
                            dst_key_1 = theme[1]
                            dst_1 = torch.LongTensor(1,1).zero_()
                            dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                            dst_1 = Variable(dst_1)
                            default_prob1 = (none_embedding(relation_dict['Theme']),-5.12)
                            for theme2 in theme2s + [('Theme2','None')] :
                                rlt_2 = torch.LongTensor(1,1).zero_()
                                rlt_2[0][0] = relation_index['Theme2']
                                rlt_2 = Variable(rlt_2)
                                dst_key_2 = theme2[1]
                                dst_2 = torch.LongTensor(1,1).zero_()
                                dst_2[0][0] = e_dict[dst_type[dst_key_2]]
                                dst_2 = Variable(dst_2)
                                default_prob2 = (none_embedding(relation_dict['Theme2']),-5.12)
                                if has_same_set({theme},event_para[src_key]) :
                                    pos_count = pos_count + 1
                                else :
                                    neg_count = neg_count + 1
                        if pos_count < neg_count :
                            min_count = pos_count
                        else :
                            min_count = neg_count
                        pos_count = min_count/pos_count
                        neg_count = min_count/neg_count
                                
                        for theme in themes :
                            rlt_1 = torch.LongTensor(1,1).zero_()
                            rlt_1[0][0] = relation_index['Theme']
                            rlt_1 = Variable(rlt_1)
                            dst_key_1 = theme[1]
                            dst_1 = torch.LongTensor(1,1).zero_()
                            dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                            dst_1 = Variable(dst_1)
                            default_prob1 = (none_embedding(relation_dict['Theme']),-5.12)
                            for theme2 in theme2s + [('Theme2','None')] :
                                rlt_2 = torch.LongTensor(1,1).zero_()
                                rlt_2[0][0] = relation_index['Theme2']
                                rlt_2 = Variable(rlt_2)
                                dst_key_2 = theme2[1]
                                dst_2 = torch.LongTensor(1,1).zero_()
                                dst_2[0][0] = e_dict[dst_type[dst_key_2]]
                                dst_2 = Variable(dst_2)
                                default_prob2 = (none_embedding(relation_dict['Theme2']),-5.12)

                                r = event_assert(src_,[(rlt_1,dst_1,rc_hidden_dict.get((src_key,dst_key_1),default_prob1)),\
                                                       (rlt_2,dst_2,rc_hidden_dict.get((src_key,dst_key_2),default_prob2))] )
                                if has_same_set({theme,theme2},event_para[src_key]) :
                                    if uniform(0,1) > pos_count :
                                        continue
                                    loss = loss + criterion_ea(r,pos_target)/loss_ea_resize
                                    loss_c = loss_c + criterion_ea(r,pos_target)/loss_ea_resize
                                    freq_dict[src_type[src_key]][0] = freq_dict[src_type[src_key]][0] + 1
                                else :
                                    if uniform(0,1) > neg_count :
                                        continue
                                    loss = loss + criterion_ea(r,neg_target)/loss_ea_resize
                                    loss_c = loss_c + criterion_ea(r,neg_target)/loss_ea_resize
                                    freq_dict[src_type[src_key]][1] = freq_dict[src_type[src_key]][1] + 1

                                if has_same_set({theme,theme2},event_para[src_key]) :
                                    pos_event_score.append(r.data[0][1])
                                else :
                                    neg_event_score.append(r.data[0][1])

                    if src_type[src_key] == 'Phosphorylation' or \
                       src_type[src_key] == 'Ubiquitination' or \
                       src_type[src_key] == 'Dephosphorylation' or \
                       src_type[src_key] == 'DNA_demethylation' or \
                       src_type[src_key] == 'Acetylation' or \
                       src_type[src_key] == 'DNA_methylation' or \
                       src_type[src_key] == 'Glycosylation' or \
                       src_type[src_key] == 'Dissociation' :
                        
                        themes = []
                        sites = []
                        for para in event_para_[src_key]:
                            if para[0] == 'Theme' :
                                themes.append(para)
                            if para[0] == 'Site' :
                                sites.append(para)

                        pos_count,neg_count = 1.0,1.0
                        for theme in themes :
                            rlt_1 = torch.LongTensor(1,1).zero_()
                            rlt_1[0][0] = relation_index['Theme']
                            rlt_1 = Variable(rlt_1)
                            dst_key_1 = theme[1]
                            dst_1 = torch.LongTensor(1,1).zero_()
                            dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                            dst_1 = Variable(dst_1)
                            default_prob1 = (none_embedding(relation_dict['Theme']),-5.12)
                            for site in sites + [('Site','None')] :
                                rlt_2 = torch.LongTensor(1,1).zero_()
                                rlt_2[0][0] = relation_index['Site']
                                rlt_2 = Variable(rlt_2)
                                dst_key_2 = site[1]
                                dst_2 = torch.LongTensor(1,1).zero_()
                                dst_2[0][0] = e_dict[dst_type[dst_key_2]]
                                dst_2 = Variable(dst_2)
                                default_prob2 = (none_embedding(relation_dict['Site']),-5.12)
                                if has_same_set({theme},event_para[src_key]) :
                                    pos_count = pos_count + 1
                                else :
                                    neg_count = neg_count + 1
                        if pos_count < neg_count :
                            min_count = pos_count
                        else :
                            min_count = neg_count
                        pos_count = min_count/pos_count
                        neg_count = min_count/neg_count
                                
                        for theme in themes :
                            rlt_1 = torch.LongTensor(1,1).zero_()
                            rlt_1[0][0] = relation_index['Theme']
                            rlt_1 = Variable(rlt_1)
                            dst_key_1 = theme[1]
                            dst_1 = torch.LongTensor(1,1).zero_()
                            dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                            dst_1 = Variable(dst_1)
                            default_prob1 = (none_embedding(relation_dict['Theme']),-5.12)
                            for site in sites + [('Site','None')] :
                                rlt_2 = torch.LongTensor(1,1).zero_()
                                rlt_2[0][0] = relation_index['Site']
                                rlt_2 = Variable(rlt_2)
                                dst_key_2 = site[1]
                                dst_2 = torch.LongTensor(1,1).zero_()
                                dst_2[0][0] = e_dict[dst_type[dst_key_2]]
                                dst_2 = Variable(dst_2)
                                default_prob2 = (none_embedding(relation_dict['Site']),-5.12)

                                r = event_assert(src_,[(rlt_1,dst_1,rc_hidden_dict.get((src_key,dst_key_1),default_prob1)),\
                                                       (rlt_2,dst_2,rc_hidden_dict.get((src_key,dst_key_2),default_prob2))] )
                                if has_same_set({theme,site},event_para[src_key]) :
                                    if uniform(0,1) > pos_count :
                                        continue
                                    loss = loss + criterion_ea(r,pos_target)/loss_ea_resize
                                    loss_c = loss_c + criterion_ea(r,pos_target)/loss_ea_resize
                                    freq_dict[src_type[src_key]][0] = freq_dict[src_type[src_key]][0] + 1
                                else :
                                    if uniform(0,1) > neg_count :
                                        continue
                                    loss = loss + criterion_ea(r,neg_target)/loss_ea_resize
                                    loss_c = loss_c + criterion_ea(r,neg_target)/loss_ea_resize
                                    freq_dict[src_type[src_key]][1] = freq_dict[src_type[src_key]][1] + 1

                                if has_same_set({theme,site},event_para[src_key]) :
                                    pos_event_score.append(r.data[0][1])
                                else :
                                    neg_event_score.append(r.data[0][1])

                    if src_type[src_key] == 'Pathway' :
                        
                        ptps = []
                        ptp2s = []
                        for para in event_para_[src_key]:
                            if para[0] == 'Participant' :
                                ptps.append(para)
                            if para[0] == 'Participant2' :
                                ptp2s.append(para)

                        pos_count,neg_count = 1.0,1.0
                        for ptp in ptps :
                            rlt_1 = torch.LongTensor(1,1).zero_()
                            rlt_1[0][0] = relation_index['Participant']
                            rlt_1 = Variable(rlt_1)
                            dst_key_1 = ptp[1]
                            dst_1 = torch.LongTensor(1,1).zero_()
                            dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                            dst_1 = Variable(dst_1)
                            default_prob1 = (none_embedding(relation_dict['Participant']),-5.12)
                            for ptp2 in ptp2s + [('Participant2','None')] :
                                rlt_2 = torch.LongTensor(1,1).zero_()
                                rlt_2[0][0] = relation_index['Participant2']
                                rlt_2 = Variable(rlt_2)
                                dst_key_2 = ptp2[1]
                                dst_2 = torch.LongTensor(1,1).zero_()
                                dst_2[0][0] = e_dict[dst_type[dst_key_2]]
                                dst_2 = Variable(dst_2)
                                default_prob2 = (none_embedding(relation_dict['Participant2']),-5.12)
                                if has_same_set({theme},event_para[src_key]) :
                                    pos_count = pos_count + 1
                                else :
                                    neg_count = neg_count + 1
                        if pos_count < neg_count :
                            min_count = pos_count
                        else :
                            min_count = neg_count
                        pos_count = min_count/pos_count
                        neg_count = min_count/neg_count
                                
                        for ptp in ptps :
                            rlt_1 = torch.LongTensor(1,1).zero_()
                            rlt_1[0][0] = relation_index['Participant']
                            rlt_1 = Variable(rlt_1)
                            dst_key_1 = ptp[1]
                            dst_1 = torch.LongTensor(1,1).zero_()
                            dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                            dst_1 = Variable(dst_1)
                            default_prob1 = (none_embedding(relation_dict['Participant']),-5.12)
                            for ptp2 in ptp2s + [('Participant2','None')] :
                                rlt_2 = torch.LongTensor(1,1).zero_()
                                rlt_2[0][0] = relation_index['Participant2']
                                rlt_2 = Variable(rlt_2)
                                dst_key_2 = ptp2[1]
                                dst_2 = torch.LongTensor(1,1).zero_()
                                dst_2[0][0] = e_dict[dst_type[dst_key_2]]
                                dst_2 = Variable(dst_2)
                                default_prob2 = (none_embedding(relation_dict['Participant2']),-5.12)

                                r = event_assert(src_,[(rlt_1,dst_1,rc_hidden_dict.get((src_key,dst_key_1),default_prob1)),\
                                                       (rlt_2,dst_2,rc_hidden_dict.get((src_key,dst_key_2),default_prob2))] )
                                if has_same_set({ptp,ptp2},event_para[src_key]) :
                                    if uniform(0,1) > pos_count :
                                        continue
                                    loss = loss + criterion_ea(r,pos_target)/loss_ea_resize
                                    loss_c = loss_c + criterion_ea(r,pos_target)/loss_ea_resize
                                    freq_dict[src_type[src_key]][0] = freq_dict[src_type[src_key]][0] + 1
                                else :
                                    if uniform(0,1) > neg_count :
                                        continue
                                    loss = loss + criterion_ea(r,neg_target)/loss_ea_resize
                                    loss_c = loss_c + criterion_ea(r,neg_target)/loss_ea_resize
                                    freq_dict[src_type[src_key]][1] = freq_dict[src_type[src_key]][1] + 1

                                if has_same_set({ptp,ptp2},event_para[src_key]) :
                                    pos_event_score.append(r.data[0][1])
                                else :
                                    neg_event_score.append(r.data[0][1])

                    if src_type[src_key] == 'Binding' :
                        themes = []
                        theme2s = []
                        sites = []
                        for para in event_para_[src_key]:
                            if para[0] == 'Theme' :
                                themes.append(para)
                            if para[0] == 'Theme2' :
                                theme2s.append(para)
                            if para[0] == 'Site' :
                                sites.append(para)

                        pos_count,neg_count = 1.0,1.0
                        for theme in themes :
                            rlt_1 = torch.LongTensor(1,1).zero_()
                            rlt_1[0][0] = relation_index['Theme']
                            rlt_1 = Variable(rlt_1)
                            dst_key_1 = theme[1]
                            dst_1 = torch.LongTensor(1,1).zero_()
                            dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                            dst_1 = Variable(dst_1)
                            default_prob1 = (none_embedding(relation_dict['Theme']),-5.12)
                            for theme2 in theme2s + [('Theme2','None')] :
                                rlt_1 = torch.LongTensor(1,1).zero_()
                                rlt_1[0][0] = relation_index['Theme2']
                                rlt_1 = Variable(rlt_1)
                                dst_key_1 = theme2[1]
                                dst_1 = torch.LongTensor(1,1).zero_()
                                dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                                dst_1 = Variable(dst_1)
                                default_prob2 = (none_embedding(relation_dict['Theme2']),-5.12)
                                for site in sites + [('Site','None')] :
                                    rlt_3 = torch.LongTensor(1,1).zero_()
                                    rlt_3[0][0] = relation_index['Site']
                                    rlt_3 = Variable(rlt_3)
                                    dst_key_3 = site[1]
                                    dst_3 = torch.LongTensor(1,1).zero_()
                                    dst_3[0][0] = e_dict[dst_type[dst_key_3]]
                                    dst_3 = Variable(dst_3)
                                    default_prob3 = (none_embedding(relation_dict['Site']),-5.12)
                                    if has_same_set({theme},event_para[src_key]) :
                                        pos_count = pos_count + 1
                                    else :
                                        neg_count = neg_count + 1
                        if pos_count < neg_count :
                            min_count = pos_count
                        else :
                            min_count = neg_count
                        pos_count = min_count/pos_count
                        neg_count = min_count/neg_count
                                
                        for theme in themes :
                            rlt_1 = torch.LongTensor(1,1).zero_()
                            rlt_1[0][0] = relation_index['Theme']
                            rlt_1 = Variable(rlt_1)
                            dst_key_1 = theme[1]
                            dst_1 = torch.LongTensor(1,1).zero_()
                            dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                            dst_1 = Variable(dst_1)
                            default_prob1 = (none_embedding(relation_dict['Theme']),-5.12)
                            for theme2 in theme2s + [('Theme2','None')] :
                                rlt_1 = torch.LongTensor(1,1).zero_()
                                rlt_1[0][0] = relation_index['Theme2']
                                rlt_1 = Variable(rlt_1)
                                dst_key_1 = theme2[1]
                                dst_1 = torch.LongTensor(1,1).zero_()
                                dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                                dst_1 = Variable(dst_1)
                                default_prob2 = (none_embedding(relation_dict['Theme2']),-5.12)
                                for site in sites + [('Site','None')] :
                                    rlt_3 = torch.LongTensor(1,1).zero_()
                                    rlt_3[0][0] = relation_index['Site']
                                    rlt_3 = Variable(rlt_3)
                                    dst_key_3 = site[1]
                                    dst_3 = torch.LongTensor(1,1).zero_()
                                    dst_3[0][0] = e_dict[dst_type[dst_key_3]]
                                    dst_3 = Variable(dst_3)
                                    default_prob3 = (none_embedding(relation_dict['Site']),-5.12)

                                    r = event_assert(src_,[(rlt_1,dst_1,rc_hidden_dict.get((src_key,dst_key_1),default_prob1)),\
                                                           (rlt_2,dst_2,rc_hidden_dict.get((src_key,dst_key_2),default_prob2)),\
                                                           (rlt_3,dst_3,rc_hidden_dict.get((src_key,dst_key_3),default_prob3))] )
                                    if has_same_set({theme,theme2,site},event_para[src_key]) :
                                        if uniform(0,1) > pos_count :
                                            continue
                                        loss = loss + criterion_ea(r,pos_target)/loss_ea_resize
                                        loss_c = loss_c + criterion_ea(r,pos_target)/loss_ea_resize
                                        freq_dict[src_type[src_key]][0] = freq_dict[src_type[src_key]][0] + 1
                                    else :
                                        if uniform(0,1) > neg_count :
                                            continue
                                        loss = loss + criterion_ea(r,neg_target)/loss_ea_resize
                                        loss_c = loss_c + criterion_ea(r,neg_target)/loss_ea_resize
                                        freq_dict[src_type[src_key]][1] = freq_dict[src_type[src_key]][1] + 1

                                    if has_same_set({theme,theme2},event_para[src_key]) :
                                        pos_event_score.append(r.data[0][1])
                                    else :
                                        neg_event_score.append(r.data[0][1])

                    if src_type[src_key] == 'Localization' :
                        
                        themes = []
                        theme2s = []
                        atlocs = []
                        fromlocs = []
                        tolocs = []
                        for para in event_para_[src_key]:
                            if para[0] == 'Theme' :
                                themes.append(para)
                            if para[0] == 'Theme2' :
                                theme2s.append(para)
                            if para[0] == 'AtLoc' :
                                atlocs.append(para)
                            if para[0] == 'FromLoc' :
                                fromlocs.append(para)
                            if para[0] == 'ToLoc' :
                                tolocs.append(para)

                        pos_count,neg_count = 1.0,1.0
                        for theme in themes :
                            rlt_1 = torch.LongTensor(1,1).zero_()
                            rlt_1[0][0] = relation_index['Theme']
                            rlt_1 = Variable(rlt_1)
                            dst_key_1 = theme[1]
                            dst_1 = torch.LongTensor(1,1).zero_()
                            dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                            dst_1 = Variable(dst_1)
                            default_prob1 = (none_embedding(relation_dict['Theme']),-5.12)
                            for theme2 in theme2s + [('Theme2','None')] :
                                rlt_2 = torch.LongTensor(1,1).zero_()
                                rlt_2[0][0] = relation_index['Theme2']
                                rlt_2 = Variable(rlt_2)
                                dst_key_2 = theme2[1]
                                dst_2 = torch.LongTensor(1,1).zero_()
                                dst_2[0][0] = e_dict[dst_type[dst_key_2]]
                                dst_2 = Variable(dst_2)
                                default_prob2 = (none_embedding(relation_dict['Theme2']),-5.12)
                                for atloc in atlocs + [('AtLoc','None')] :
                                    rlt_3 = torch.LongTensor(1,1).zero_()
                                    rlt_3[0][0] = relation_index['AtLoc']
                                    rlt_3 = Variable(rlt_3)
                                    dst_key_3 = atloc[1]
                                    dst_3 = torch.LongTensor(1,1).zero_()
                                    dst_3[0][0] = e_dict[dst_type[dst_key_3]]
                                    dst_3 = Variable(dst_3)
                                    default_prob3 = (none_embedding(relation_dict['AtLoc']),-5.12)
                                    for fromloc in fromlocs + [('FromLoc','None')] :
                                        rlt_4 = torch.LongTensor(1,1).zero_()
                                        rlt_4[0][0] = relation_index['FromLoc']
                                        rlt_4 = Variable(rlt_4)
                                        dst_key_4 = fromloc[1]
                                        dst_4 = torch.LongTensor(1,1).zero_()
                                        dst_4[0][0] = e_dict[dst_type[dst_key_4]]
                                        dst_4 = Variable(dst_4)
                                        default_prob4 = (none_embedding(relation_dict['FromLoc']),-5.12)
                                        for toloc in tolocs + [('ToLoc','None')] :
                                            rlt_5 = torch.LongTensor(1,1).zero_()
                                            rlt_5[0][0] = relation_index['ToLoc']
                                            rlt_5 = Variable(rlt_5)
                                            dst_key_5 = toloc[1]
                                            dst_5 = torch.LongTensor(1,1).zero_()
                                            dst_5[0][0] = e_dict[dst_type[dst_key_5]]
                                            dst_5 = Variable(dst_5)
                                            default_prob5 = (none_embedding(relation_dict['ToLoc']),-5.12)
                                            if has_same_set({theme},event_para[src_key]) :
                                                pos_count = pos_count + 1
                                            else :
                                                neg_count = neg_count + 1
                        if pos_count < neg_count :
                            min_count = pos_count
                        else :
                            min_count = neg_count
                        pos_count = min_count/pos_count
                        neg_count = min_count/neg_count

                        for theme in themes :
                            rlt_1 = torch.LongTensor(1,1).zero_()
                            rlt_1[0][0] = relation_index['Theme']
                            rlt_1 = Variable(rlt_1)
                            dst_key_1 = theme[1]
                            dst_1 = torch.LongTensor(1,1).zero_()
                            dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                            dst_1 = Variable(dst_1)
                            default_prob1 = (none_embedding(relation_dict['Theme']),-5.12)
                            for theme2 in theme2s + [('Theme2','None')] :
                                rlt_2 = torch.LongTensor(1,1).zero_()
                                rlt_2[0][0] = relation_index['Theme2']
                                rlt_2 = Variable(rlt_2)
                                dst_key_2 = theme2[1]
                                dst_2 = torch.LongTensor(1,1).zero_()
                                dst_2[0][0] = e_dict[dst_type[dst_key_2]]
                                dst_2 = Variable(dst_2)
                                default_prob2 = (none_embedding(relation_dict['Theme2']),-5.12)
                                for atloc in atlocs + [('AtLoc','None')] :
                                    rlt_3 = torch.LongTensor(1,1).zero_()
                                    rlt_3[0][0] = relation_index['AtLoc']
                                    rlt_3 = Variable(rlt_3)
                                    dst_key_3 = atloc[1]
                                    dst_3 = torch.LongTensor(1,1).zero_()
                                    dst_3[0][0] = e_dict[dst_type[dst_key_3]]
                                    dst_3 = Variable(dst_3)
                                    default_prob3 = (none_embedding(relation_dict['AtLoc']),-5.12)
                                    for fromloc in fromlocs + [('FromLoc','None')] :
                                        rlt_4 = torch.LongTensor(1,1).zero_()
                                        rlt_4[0][0] = relation_index['FromLoc']
                                        rlt_4 = Variable(rlt_4)
                                        dst_key_4 = fromloc[1]
                                        dst_4 = torch.LongTensor(1,1).zero_()
                                        dst_4[0][0] = e_dict[dst_type[dst_key_4]]
                                        dst_4 = Variable(dst_4)
                                        default_prob4 = (none_embedding(relation_dict['FromLoc']),-5.12)
                                        for toloc in tolocs + [('ToLoc','None')] :
                                            rlt_5 = torch.LongTensor(1,1).zero_()
                                            rlt_5[0][0] = relation_index['ToLoc']
                                            rlt_5 = Variable(rlt_5)
                                            dst_key_5 = toloc[1]
                                            dst_5 = torch.LongTensor(1,1).zero_()
                                            dst_5[0][0] = e_dict[dst_type[dst_key_5]]
                                            dst_5 = Variable(dst_5)
                                            default_prob5 = (none_embedding(relation_dict['ToLoc']),-5.12)
                                            r = event_assert(src_,[(rlt_1,dst_1,rc_hidden_dict.get((src_key,dst_key_1),default_prob1)),\
                                                                   (rlt_2,dst_2,rc_hidden_dict.get((src_key,dst_key_2),default_prob2)),\
                                                                   (rlt_3,dst_3,rc_hidden_dict.get((src_key,dst_key_3),default_prob3)),\
                                                                   (rlt_4,dst_4,rc_hidden_dict.get((src_key,dst_key_4),default_prob4)),\
                                                                   (rlt_5,dst_5,rc_hidden_dict.get((src_key,dst_key_5),default_prob5))] )        
                                            if has_same_set({theme,theme2,atloc,fromloc,toloc},event_para[src_key]) :
                                                if uniform(0,1) > pos_count :
                                                    continue
                                                loss = loss + criterion_ea(r,pos_target)/loss_ea_resize
                                                loss_c = loss_c + criterion_ea(r,pos_target)/loss_ea_resize
                                                freq_dict[src_type[src_key]][0] = freq_dict[src_type[src_key]][0] + 1
                                            else :
                                                if uniform(0,1) > neg_count :
                                                    continue
                                                loss = loss + criterion_ea(r,neg_target)/loss_ea_resize
                                                loss_c = loss_c + criterion_ea(r,neg_target)/loss_ea_resize
                                                freq_dict[src_type[src_key]][1] = freq_dict[src_type[src_key]][1] + 1

                                            if has_same_set({theme,theme2,atloc,fromloc,toloc},event_para[src_key]) :
                                                pos_event_score.append(r.data[0][1])
                                            else :
                                                neg_event_score.append(r.data[0][1])

                    if src_type[src_key] == 'Regulation' or \
                       src_type[src_key] == 'Negative_regulation' or \
                       src_type[src_key] == 'Positive_regulation' :
                        
                        themes = []
                        causes = []
                        for para in event_para_[src_key]:
                            if para[0] == 'Theme' :
                                themes.append(para)
                            if para[0] == 'Cause' :
                                causes.append(para)

                        pos_count,neg_count = 1.0,1.0
                        for theme in themes :
                            rlt_1 = torch.LongTensor(1,1).zero_()
                            rlt_1[0][0] = relation_index['Theme']
                            rlt_1 = Variable(rlt_1)
                            dst_key_1 = theme[1]
                            dst_1 = torch.LongTensor(1,1).zero_()
                            dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                            dst_1 = Variable(dst_1)
                            default_prob1 = (none_embedding(relation_dict['Theme']),-5.12)
                            for cause in causes + [('Cause','None')] :
                                rlt_2 = torch.LongTensor(1,1).zero_()
                                rlt_2[0][0] = relation_index['Cause']
                                rlt_2 = Variable(rlt_2)
                                dst_key_2 = cause[1]
                                dst_2 = torch.LongTensor(1,1).zero_()
                                dst_2[0][0] = e_dict[dst_type[dst_key_2]]
                                dst_2 = Variable(dst_2)
                                default_prob2 = (none_embedding(relation_dict['Cause']),-5.12)
                                if has_same_set({theme},event_para[src_key]) :
                                    pos_count = pos_count + 1
                                else :
                                    neg_count = neg_count + 1
                        if pos_count < neg_count :
                            min_count = pos_count
                        else :
                            min_count = neg_count
                        pos_count = min_count/pos_count
                        neg_count = min_count/neg_count
                                
                        for theme in themes :
                            rlt_1 = torch.LongTensor(1,1).zero_()
                            rlt_1[0][0] = relation_index['Theme']
                            rlt_1 = Variable(rlt_1)
                            dst_key_1 = theme[1]
                            dst_1 = torch.LongTensor(1,1).zero_()
                            dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                            dst_1 = Variable(dst_1)
                            default_prob1 = (none_embedding(relation_dict['Theme']),-5.12)
                            for cause in causes + [('Cause','None')] :
                                rlt_2 = torch.LongTensor(1,1).zero_()
                                rlt_2[0][0] = relation_index['Cause']
                                rlt_2 = Variable(rlt_2)
                                dst_key_2 = cause[1]
                                dst_2 = torch.LongTensor(1,1).zero_()
                                dst_2[0][0] = e_dict[dst_type[dst_key_2]]
                                dst_2 = Variable(dst_2)
                                default_prob2 = (none_embedding(relation_dict['Cause']),-5.12)

                                r = event_assert(src_,[(rlt_1,dst_1,rc_hidden_dict.get((src_key,dst_key_1),default_prob1)),\
                                                       (rlt_2,dst_2,rc_hidden_dict.get((src_key,dst_key_2),default_prob2))] )
                                if has_same_set({theme,cause},event_para[src_key]) :
                                    if uniform(0,1) > pos_count :
                                        continue
                                    loss = loss + criterion_ea(r,pos_target)/loss_ea_resize
                                    loss_c = loss_c + criterion_ea(r,pos_target)/loss_ea_resize
                                    freq_dict[src_type[src_key]][0] = freq_dict[src_type[src_key]][0] + 1
                                else :
                                    if uniform(0,1) > neg_count :
                                        continue
                                    loss = loss + criterion_ea(r,neg_target)/loss_ea_resize
                                    loss_c = loss_c + criterion_ea(r,neg_target)/loss_ea_resize
                                    freq_dict[src_type[src_key]][1] = freq_dict[src_type[src_key]][1] + 1

                                if has_same_set({theme,cause},event_para[src_key]) :
                                    pos_event_score.append(r.data[0][1])
                                else :
                                    neg_event_score.append(r.data[0][1])

                    if src_type[src_key] == 'Planned_process' :
                        themes = []
                        theme2s = []
                        instrs = []
                        instr2s = []
                        for para in event_para_[src_key]:
                            if para[0] == 'Theme' :
                                themes.append(para)
                            if para[0] == 'Theme2' :
                                theme2s.append(para)
                            if para[0] == 'Instrument' :
                                instrs.append(para)
                            if para[0] == 'Instrument2' :
                                instr2s.append(para)

                        pos_count,neg_count = 1.0,1.0
                        for theme in themes + [('Theme','None')] :
                            rlt_1 = torch.LongTensor(1,1).zero_()
                            rlt_1[0][0] = relation_index['Theme']
                            rlt_1 = Variable(rlt_1)
                            dst_key_1 = theme[1]
                            dst_1 = torch.LongTensor(1,1).zero_()
                            dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                            dst_1 = Variable(dst_1)
                            default_prob1 = (none_embedding(relation_dict['Theme']),-5.12)
                            for theme2 in theme2s + [('Theme2','None')] :
                                rlt_2 = torch.LongTensor(1,1).zero_()
                                rlt_2[0][0] = relation_index['Theme2']
                                rlt_2 = Variable(rlt_2)
                                dst_key_2 = theme2[1]
                                dst_2 = torch.LongTensor(1,1).zero_()
                                dst_2[0][0] = e_dict[dst_type[dst_key_2]]
                                dst_2 = Variable(dst_2)
                                default_prob2 = (none_embedding(relation_dict['Theme2']),-5.12)
                                for instr in instrs + [('Instrument','None')] :
                                    rlt_3 = torch.LongTensor(1,1).zero_()
                                    rlt_3[0][0] = relation_index['Instrument']
                                    rlt_3 = Variable(rlt_3)
                                    dst_key_3 = instr[1]
                                    dst_3 = torch.LongTensor(1,1).zero_()
                                    dst_3[0][0] = e_dict[dst_type[dst_key_3]]
                                    dst_3 = Variable(dst_3)
                                    default_prob3 = (none_embedding(relation_dict['Instrument']),-5.12)
                                    for instr2 in instr2s + [('Instrument2','None')] :
                                        rlt_4 = torch.LongTensor(1,1).zero_()
                                        rlt_4[0][0] = relation_index['Instrument2']
                                        rlt_4 = Variable(rlt_4)
                                        dst_key_4 = instr2[1]
                                        dst_4 = torch.LongTensor(1,1).zero_()
                                        dst_4[0][0] = e_dict[dst_type[dst_key_4]]
                                        dst_4 = Variable(dst_4)
                                        default_prob4 = (none_embedding(relation_dict['Instrument2']),-5.12)
                                        if has_same_set({theme},event_para[src_key]) :
                                            pos_count = pos_count + 1
                                        else :
                                            neg_count = neg_count + 1
                        if pos_count < neg_count :
                            min_count = pos_count
                        else :
                            min_count = neg_count
                        pos_count = min_count/pos_count
                        neg_count = min_count/neg_count

                        for theme in themes + [('Theme','None')] :
                            rlt_1 = torch.LongTensor(1,1).zero_()
                            rlt_1[0][0] = relation_index['Theme']
                            rlt_1 = Variable(rlt_1)
                            dst_key_1 = theme[1]
                            dst_1 = torch.LongTensor(1,1).zero_()
                            dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                            dst_1 = Variable(dst_1)
                            default_prob1 = (none_embedding(relation_dict['Theme']),-5.12)
                            for theme2 in theme2s + [('Theme2','None')] :
                                rlt_2 = torch.LongTensor(1,1).zero_()
                                rlt_2[0][0] = relation_index['Theme2']
                                rlt_2 = Variable(rlt_2)
                                dst_key_2 = theme2[1]
                                dst_2 = torch.LongTensor(1,1).zero_()
                                dst_2[0][0] = e_dict[dst_type[dst_key_2]]
                                dst_2 = Variable(dst_2)
                                default_prob2 = (none_embedding(relation_dict['Theme2']),-5.12)
                                for instr in instrs + [('Instrument','None')] :
                                    rlt_3 = torch.LongTensor(1,1).zero_()
                                    rlt_3[0][0] = relation_index['Instrument']
                                    rlt_3 = Variable(rlt_3)
                                    dst_key_3 = instr[1]
                                    dst_3 = torch.LongTensor(1,1).zero_()
                                    dst_3[0][0] = e_dict[dst_type[dst_key_3]]
                                    dst_3 = Variable(dst_3)
                                    default_prob3 = (none_embedding(relation_dict['Instrument']),-5.12)
                                    for instr2 in instr2s + [('Instrument2','None')] :
                                        rlt_4 = torch.LongTensor(1,1).zero_()
                                        rlt_4[0][0] = relation_index['Instrument2']
                                        rlt_4 = Variable(rlt_4)
                                        dst_key_4 = instr2[1]
                                        dst_4 = torch.LongTensor(1,1).zero_()
                                        dst_4[0][0] = e_dict[dst_type[dst_key_4]]
                                        dst_4 = Variable(dst_4)
                                        default_prob4 = (none_embedding(relation_dict['Instrument2']),-5.12)

                                        r = event_assert(src_,[(rlt_1,dst_1,rc_hidden_dict.get((src_key,dst_key_1),default_prob1)),\
                                                               (rlt_2,dst_2,rc_hidden_dict.get((src_key,dst_key_2),default_prob2)),\
                                                               (rlt_3,dst_3,rc_hidden_dict.get((src_key,dst_key_3),default_prob3)),\
                                                               (rlt_4,dst_4,rc_hidden_dict.get((src_key,dst_key_4),default_prob4))] )
                                        if has_same_set({theme,theme2,instr,instr2},event_para[src_key]) :
                                            if uniform(0,1) > pos_count :
                                                continue
                                            loss = loss + criterion_ea(r,pos_target)/loss_ea_resize
                                            loss_c = loss_c + criterion_ea(r,pos_target)/loss_ea_resize
                                            freq_dict[src_type[src_key]][0] = freq_dict[src_type[src_key]][0] + 1
                                        else :
                                            if uniform(0,1) > neg_count :
                                                continue
                                            loss = loss + criterion_ea(r,neg_target)/loss_ea_resize
                                            loss_c = loss_c + criterion_ea(r,neg_target)/loss_ea_resize
                                            freq_dict[src_type[src_key]][1] = freq_dict[src_type[src_key]][1] + 1
                                
                                        if has_same_set({theme,theme2,instr,instr2},event_para[src_key]) :
                                            pos_event_score.append(r.data[0][1])
                                        else :
                                            neg_event_score.append(r.data[0][1])
                                
                ############################################################
	    loss.backward()
	    optimizer.step()

            running_loss += loss.data[0]
            if i%10 == 9:
                print('[%d, %3d] loss: %.5f %.5f %.5f %.5f' % (epoch+1,i+1,running_loss/80,loss_a/80,loss_b/80,loss_c/80)) # step is 10 and batch_size is 8
                #print valid_count/(valid_count+invalid_count)
                running_loss = 0.0
                loss_a,loss_b,loss_c = 0.0, 0.0, 0.0
                valid_count, invalid_count = 0.0, 0.0
                
        '''        
        test_loss,test_loss_a,test_loss_b = testset_loss(testset,char_cnn,bilstm,ner,rc,criterion_ner,criterion_rc)
        print('[%d ] test loss: %.5f %.5f %.5f' % (epoch+1,test_loss,test_loss_a,test_loss_b))
        '''
        print len(pos_event_score)
        print sum(pos_event_score)/len(pos_event_score)
        print len(neg_event_score)
        print sum(neg_event_score)/len(neg_event_score)
        
        if (epoch+1) % 10 == 0:
            for key in freq_dict :
                if freq_dict[key][0]+freq_dict[key][1] == 0 :
                    continue
                print key,'%d %d %.5f'%(freq_dict[key][0],freq_dict[key][1],float(freq_dict[key][0])/(freq_dict[key][0]+freq_dict[key][1]))
                
            torch.save(char_cnn.state_dict(),path_+'/char_cnn_%d_%d.pth'%(epoch+1,number))
            torch.save(bilstm.state_dict(),path_+'/bilstm_%d_%d.pth'%(epoch+1,number))
            torch.save(ner.state_dict(),path_+'/ner_%d_%d.pth'%(epoch+1,number))
            torch.save(rc.state_dict(),path_+'/rc_%d_%d.pth'%(epoch+1,number))
            torch.save(none_embedding.state_dict(),path_+'/none_embedding_%d_%d.pth'%(epoch+1,number))
            torch.save(event_assert.state_dict(),path_+'/assert_nets/event_assert_%d_%d.pth'%(epoch+1,number))
            
    print('Finished Training')
