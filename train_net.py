import os
import sys
import torch
import gensim
import cPickle
import collections
import numpy as np
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.init as init
from random import uniform
from torch.autograd import Variable
from sentence_set import Sentence_Set
from dataloader_modified import DataLoader
from define_net import Char_CNN_pretrain,Char_CNN_encode,BiLSTM,Trigger_Recognition,Relation_Classification
from define_net_event_evaluation import *
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
        
    for key in subsetb.keys() :
        value = subsetb[key]
        if is_same_set(seta,value) :
            return True,key
    
    return False,'None'


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

    f = file(path_+'/word_index_all', 'r')
    word_index = cPickle.load(f)

    trainset = Sentence_Set(path_+'/table/',new_dict=False)
    #testset = Sentence_Set(path_+'/table_test/',new_dict=False)
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
    trainloader = DataLoader(trainset,batch_size=16,shuffle=True)

    char_cnn = Char_CNN_encode(char_dim).cuda()
    bilstm = BiLSTM(word_dim,entity_dim).cuda()
    tr = Trigger_Recognition(event_dim).cuda()
    rc = Relation_Classification(relation_dim).cuda()
    ee = Event_Evaluation().cuda()

    ccp = Char_CNN_pretrain(char_dim,event_dim)
    ccp.load_state_dict(torch.load(path_+'/nets/char_rnn_pretrain.pth'))
    new_dict = collections.OrderedDict()
    new_dict['embedding.weight'] = ccp.state_dict()['embedding.weight']
    new_dict['conv.weight'] = ccp.state_dict()['conv.weight']
    new_dict['conv.bias'] = ccp.state_dict()['conv.bias']
    char_cnn.load_state_dict(new_dict)
    '''
    for p in char_cnn.embedding.parameters():
	p.requires_grad = False
    for p in char_cnn.conv.parameters():
	p.requires_grad = False
    '''
    init.xavier_uniform(char_cnn.embedding.weight)
    init.xavier_uniform(char_cnn.conv.weight)
    init.xavier_uniform(bilstm.word_embedding.weight)
    init.xavier_uniform(bilstm.entity_embedding.weight)
    init.xavier_uniform(bilstm.lstm.weight_ih_l0)
    init.xavier_uniform(bilstm.lstm.weight_hh_l0)
    init.xavier_uniform(bilstm.lstm.weight_ih_l0_reverse)
    init.xavier_uniform(bilstm.lstm.weight_hh_l0_reverse)
    init.xavier_uniform(tr.event_embedding.weight)
    init.xavier_uniform(tr.linear.weight)
    init.xavier_uniform(tr.linear2.weight)
    init.xavier_uniform(rc.linear.weight)
    init.xavier_uniform(rc.linear2.weight)
    init.xavier_uniform(rc.empty_embedding.weight)
    init.xavier_uniform(rc.position_embedding.weight)
    init.xavier_uniform(rc.conv_3.weight)
    init.xavier_uniform(rc.conv_3r.weight)
    init.xavier_uniform(ee.role_embedding.weight)
    init.xavier_uniform(ee.lstm.weight_ih_l0)
    init.xavier_uniform(ee.lstm.weight_hh_l0)
    init.xavier_uniform(ee.lstm.weight_ih_l0_reverse)
    init.xavier_uniform(ee.lstm.weight_hh_l0_reverse)
    init.xavier_uniform(ee.linear1.weight)
    init.xavier_uniform(ee.linear2.weight)
    init.xavier_uniform(ee.linearm_1.weight)
    init.xavier_uniform(ee.linearm_2.weight)
    
    print char_cnn
    print bilstm
    print tr
    print rc
    print ee

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
    word_embedding = bilstm.state_dict()['word_embedding.weight'].cpu().numpy()
    word_embedding = load_pretrain_vector(word_index,word_embedding)
    pretrained_weight = np.array(word_embedding)
    bilstm.word_embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))   
    '''
    class_weight_tr = [ 5 for i in range(0,event_dim) ] #
    class_weight_tr[0] = 1
    class_weight_tr = torch.FloatTensor(class_weight_tr).cuda()
    criterion_tr = nn.NLLLoss(weight=class_weight_tr)
    
    class_weight_rc = [ 5 for i in range(0,relation_dim) ] #
    class_weight_rc[2] = 1
    class_weight_rc = torch.FloatTensor(class_weight_rc).cuda()
    criterion_rc = nn.NLLLoss(weight=class_weight_rc)
    
    class_weight_ee = [1,1] #
    class_weight_ee = torch.FloatTensor(class_weight_ee).cuda()
    criterion_ee = nn.NLLLoss(weight=class_weight_ee)

    class_weight_m = [1,5,5] #
    class_weight_m = torch.FloatTensor(class_weight_m).cuda()
    criterion_m = nn.NLLLoss(weight=class_weight_m) 

    optimizer = optim.Adam(list(char_cnn.parameters())+\
                           list(bilstm.parameters())+\
                           list(tr.parameters())+\
                           list(rc.parameters())+\
                           list(ee.parameters()), lr=0.007, weight_decay=0.0002)

    pos_target = Variable(torch.LongTensor([1])).cuda()
    neg_target = Variable(torch.LongTensor([0])).cuda()

    modification_target = dict()
    modification_target['None'] = Variable(torch.LongTensor([0])).cuda()
    modification_target['Speculation'] = Variable(torch.LongTensor([1])).cuda()
    modification_target['Negation'] = Variable(torch.LongTensor([2])).cuda()
    
    for epoch in range(50): #

        running_loss = 0.0
        loss_a,loss_b,loss_c,loss_d = 0.0,0.0,0.0,0.0
        addition_count = 0.0

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

                input_word, input_entity, input_char, target, entity_loc, event_loc, relation, event_para, modification, fname = data
                input_word, input_entity, target = Variable(input_word).cuda(),Variable(input_entity).cuda(),Variable(target).cuda()
                
                char_encode = []
                for chars in input_char :
                    chars = char_cnn(Variable(chars).cuda())
                    char_encode.append(chars)
                char_encode = torch.cat(char_encode,0) # L*N*conv_out_channel

                hidden = bilstm.initHidden()
		bilstm_output,hidden,entity_emb = bilstm((input_word,input_entity,char_encode),hidden)
		tr_output,tr_emb,tr_index = tr(bilstm_output)
                
		target = target.view(-1) # L of indices
		loss = loss + criterion_tr(tr_output,target) # NLLloss only accept 1-dimension
                loss_a = loss_a + criterion_tr(tr_output,target)

                for j in range(0,target.size()[0]):
                    if int(target[j]) != int(tr_index[j]) :
                        vector = tr.get_event_embedding(int(target[j]))
                        tr_emb[j] = vector
                
                e_loc = dict(entity_loc.items()+event_loc.items())

                event_para_ = dict()
                src_type = dict()
                dst_type = dict()

                for rlt in relation.keys() :
                    
                    src_key = rlt[0]
                    dst_key = rlt[1]
                    rlt_target = Variable(relation[rlt]).cuda()
                    src_range = e_loc[src_key]
                    dst_range = e_loc[dst_key]
                    
                    src_begin,src_end = src_range
                    dst_begin,dst_end = dst_range
                    # if the prediction of event is incorrect, ignore them
                    correct_event = True
                    for j in range(src_begin,src_end+1):
                        if int(target[j]) != int(tr_index[j]) :
                            correct_event = False
                    if event_loc.has_key(dst_key):
                        for j in range(dst_begin,dst_end+1):
                            if int(target[j]) != int(tr_index[j]) :
                                correct_event = False
                    '''
                    if not correct_event :
                        continue
                    '''
                    src = bilstm_output[src_begin:src_end+1]
                    src_event = tr_emb[src_begin:src_end+1]
                    dst = bilstm_output[dst_begin:dst_end+1]

                    src_name = event_index_r[int(target[src_begin])].split('-')[0]
                    src_type[src_key] = src_name
                    
                    event_flag = False
                    if event_loc.has_key(dst_key):
                        dst_event = tr_emb[dst_begin:dst_end+1]
                        dst_name = event_index_r[int(target[dst_begin])].split('-')[0]
                        event_flag = True
                    else :
                        dst_event = entity_emb[dst_begin:dst_end+1]
                        dst_name = entity_index_r[int(input_entity.data[0][dst_begin])].split('-')[0]
                    dst_type[dst_key] = dst_name

                    if event_flag and not src_name in {'Regulation','Positive_regulation','Negative_regulation','Planned_process'} :
                        continue
                    
                    reverse_flag = False
                    if src_begin > dst_begin :
                        reverse_flag = True
                        
                    middle_flag = False
                    if  src_end+1 < dst_begin:
                        middle = bilstm_output[src_end+1:dst_begin]
                    elif dst_end < src_begin-1:
                        middle = bilstm_output[dst_end+1:src_begin]
                    else : # adjacent or overlapped
                        middle_flag = True
                        middle = Variable(torch.zeros(1,1,128*2)).cuda() # L(=1)*N*2 self.hidden_size
                            
                    rc_output = rc(src_event,src,middle,dst,dst_event,reverse_flag,middle_flag)
                              
                    loss = loss + criterion_rc(rc_output,rlt_target)/10.0
                    loss_b = loss_b + criterion_rc(rc_output,rlt_target)/10.0

                    row = rc_output.data
                    this_row = list(row[0])
                    index = this_row.index(max(this_row))
                    current_type = relation_index_r[index]
                    '''
                    if relation[(src_key,dst_key)][0] != index :
                        continue
                    '''
                    correct_index = relation[(src_key,dst_key)][0]
                    current_type = relation_index_r[int(correct_index)]
                    
                    if current_type != 'NONE' :
                        if src_key not in event_para.keys() :
                            continue # detail in notebook
                        if not event_para_.has_key(src_key) :
                            event_para_[src_key] = set()
                            event_para_[src_key].add((current_type,dst_key))
                        else :
                            event_para_[src_key].add((current_type,dst_key))

                dst_type['None'] = 'None'
                e_loc['None'] = (0,0)
                
                loss_ee_resize = 2.0
                ############################################################
                for src_key in event_para_.keys() :

                    s_r = e_loc[src_key]
                    range_list = []
                    pos_count,neg_count = 1.0,1.0
                    
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

                        for theme in themes :
                            if has_same_set({theme},event_para[src_key])[0] :
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
                            dst_key = theme[1]
                            range_list.append(('Theme',e_loc[dst_key]))
                            
                            hidden = ee.initHidden()
                            r,m = ee(bilstm_output,tr_emb,entity_emb,s_r,range_list,hidden)
                            e_flag,e_id = has_same_set({theme},event_para[src_key])
                            
                            if e_flag :
                                if uniform(0,1) > pos_count :
                                    continue
                                loss = loss + criterion_ee(r,pos_target)/loss_ee_resize
                                loss_c = loss_c + criterion_ee(r,pos_target)/loss_ee_resize
                                m_type = modification.get(e_id,'None')
                                m_target = modification_target[m_type]
                                loss = loss + criterion_m(m,m_target)
                                loss_d = loss_d + criterion_m(m,m_target)
                                freq_dict[src_type[src_key]][0] = freq_dict[src_type[src_key]][0] + 1
                            else :
                                if uniform(0,1) > neg_count :
                                    continue
                                loss = loss + criterion_ee(r,neg_target)/loss_ee_resize
                                loss_c = loss_c + criterion_ee(r,neg_target)/loss_ee_resize
                                freq_dict[src_type[src_key]][1] = freq_dict[src_type[src_key]][1] + 1

                            range_list.pop()
                                
                    if src_type[src_key] == 'Cell_death' or \
                       src_type[src_key] == 'Amino_acid_catabolism' or \
                       src_type[src_key] == 'Glycolysis':
                        
                        themes = []
                        for para in event_para_[src_key]:
                            if para[0] == 'Theme' :
                                themes.append(para)

                        for theme in themes + [('Theme','None')] :
                            if has_same_set({theme},event_para[src_key])[0] :
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
                            dst_key = theme[1]
                            range_list.append(('Theme',e_loc[dst_key]))
                            
                            hidden = ee.initHidden()
                            r,m = ee(bilstm_output,tr_emb,entity_emb,s_r,range_list,hidden)
                            e_flag,e_id = has_same_set({theme},event_para[src_key])

                            if e_flag :
                                if uniform(0,1) > pos_count :
                                    continue
                                loss = loss + criterion_ee(r,pos_target)/loss_ee_resize
                                loss_c = loss_c + criterion_ee(r,pos_target)/loss_ee_resize
                                m_type = modification.get(e_id,'None')
                                m_target = modification_target[m_type]
                                loss = loss + criterion_m(m,m_target)
                                loss_d = loss_d + criterion_m(m,m_target)
                                freq_dict[src_type[src_key]][0] = freq_dict[src_type[src_key]][0] + 1
                            else :
                                if uniform(0,1) > neg_count :
                                    continue
                                loss = loss + criterion_ee(r,neg_target)/loss_ee_resize
                                loss_c = loss_c + criterion_ee(r,neg_target)/loss_ee_resize
                                freq_dict[src_type[src_key]][1] = freq_dict[src_type[src_key]][1] + 1

                            range_list.pop()

                    if src_type[src_key] == 'Cell_differentiation' :
                        
                        themes = []
                        atlocs = []
                        for para in event_para_[src_key]:
                            if para[0] == 'Theme' :
                                themes.append(para)
                            if para[0] == 'AtLoc' :
                                atlocs.append(para)
                        
                        for theme in themes :
                            for atloc in atlocs + [('AtLoc','None')] :
                                if has_same_set({theme,atloc},event_para[src_key])[0] :
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
                            dst_key = theme[1]
                            range_list.append(('Theme',e_loc[dst_key]))
                            
                            for atloc in atlocs + [('AtLoc','None')] :
                                dst_key = atloc[1]
                                range_list.append(('AtLoc',e_loc[dst_key]))

                                hidden = ee.initHidden()
                                r,m = ee(bilstm_output,tr_emb,entity_emb,s_r,range_list,hidden)
                                e_flag,e_id = has_same_set({theme,atloc},event_para[src_key])
                            
                                if e_flag :
                                    if uniform(0,1) > pos_count :
                                        continue
                                    loss = loss + criterion_ee(r,pos_target)/loss_ee_resize
                                    loss_c = loss_c + criterion_ee(r,pos_target)/loss_ee_resize
                                    m_type = modification.get(e_id,'None')
                                    m_target = modification_target[m_type]
                                    loss = loss + criterion_m(m,m_target)
                                    loss_d = loss_d + criterion_m(m,m_target)
                                    freq_dict[src_type[src_key]][0] = freq_dict[src_type[src_key]][0] + 1
                                else :
                                    if uniform(0,1) > neg_count :
                                        continue
                                    loss = loss + criterion_ee(r,neg_target)/loss_ee_resize
                                    loss_c = loss_c + criterion_ee(r,neg_target)/loss_ee_resize
                                    freq_dict[src_type[src_key]][1] = freq_dict[src_type[src_key]][1] + 1

                                range_list.pop()
                            range_list.pop()

                    if src_type[src_key] == 'Blood_vessel_development' or \
                       src_type[src_key] == 'Carcinogenesis'or \
                       src_type[src_key] == 'Cell_transformation' :
                        
                        themes = []
                        atlocs = []
                        for para in event_para_[src_key]:
                            if para[0] == 'Theme' :
                                themes.append(para)
                            if para[0] == 'AtLoc' :
                                atlocs.append(para)

                        for theme in themes + [('Theme','None')] :
                            for atloc in atlocs + [('AtLoc','None')] :
                                if has_same_set({theme,atloc},event_para[src_key])[0] :
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
                            dst_key = theme[1]
                            range_list.append(('Theme',e_loc[dst_key]))
                            
                            for atloc in atlocs + [('AtLoc','None')] :
                                dst_key = atloc[1]
                                range_list.append(('AtLoc',e_loc[dst_key]))

                                hidden = ee.initHidden()
                                r,m = ee(bilstm_output,tr_emb,entity_emb,s_r,range_list,hidden)
                                e_flag,e_id = has_same_set({theme,atloc},event_para[src_key])

                                if e_flag :
                                    if uniform(0,1) > pos_count :
                                        continue
                                    loss = loss + criterion_ee(r,pos_target)/loss_ee_resize
                                    loss_c = loss_c + criterion_ee(r,pos_target)/loss_ee_resize
                                    m_type = modification.get(e_id,'None')
                                    m_target = modification_target[m_type]
                                    loss = loss + criterion_m(m,m_target)
                                    loss_d = loss_d + criterion_m(m,m_target)
                                    freq_dict[src_type[src_key]][0] = freq_dict[src_type[src_key]][0] + 1
                                else :
                                    if uniform(0,1) > neg_count :
                                        continue
                                    loss = loss + criterion_ee(r,neg_target)/loss_ee_resize
                                    loss_c = loss_c + criterion_ee(r,neg_target)/loss_ee_resize
                                    freq_dict[src_type[src_key]][1] = freq_dict[src_type[src_key]][1] + 1

                                range_list.pop()
                            range_list.pop()

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

                        for theme in themes + [('Theme','None')] :
                            for atloc in atlocs + [('AtLoc','None')] :
                                for site in sites + [('Site','None')] :
                                    if has_same_set({theme,atloc,site},event_para[src_key])[0] :
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
                            dst_key = theme[1]
                            range_list.append(('Theme',e_loc[dst_key]))
                            
                            for atloc in atlocs + [('AtLoc','None')] :
                                dst_key = atloc[1]
                                range_list.append(('AtLoc',e_loc[dst_key]))
                            
                                for site in sites + [('Site','None')] :
                                    dst_key = site[1]
                                    range_list.append(('Site',e_loc[dst_key]))

                                    hidden = ee.initHidden()
                                    r,m = ee(bilstm_output,tr_emb,entity_emb,s_r,range_list,hidden)
                                    e_flag,e_id = has_same_set({theme,atloc,site},event_para[src_key])
                            
                                    if e_flag :
                                        if uniform(0,1) > pos_count :
                                            continue
                                        loss = loss + criterion_ee(r,pos_target)/loss_ee_resize
                                        loss_c = loss_c + criterion_ee(r,pos_target)/loss_ee_resize
                                        m_type = modification.get(e_id,'None')
                                        m_target = modification_target[m_type]
                                        loss = loss + criterion_m(m,m_target)
                                        loss_d = loss_d + criterion_m(m,m_target)
                                        freq_dict[src_type[src_key]][0] = freq_dict[src_type[src_key]][0] + 1
                                    else :
                                        if uniform(0,1) > neg_count :
                                            continue
                                        loss = loss + criterion_ee(r,neg_target)/loss_ee_resize
                                        loss_c = loss_c + criterion_ee(r,neg_target)/loss_ee_resize
                                        freq_dict[src_type[src_key]][1] = freq_dict[src_type[src_key]][1] + 1

                                    range_list.pop()
                                range_list.pop()
                            range_list.pop()

                    if src_type[src_key] == 'Metastasis' :
                        themes = []
                        tolocs = []
                        for para in event_para_[src_key]:
                            if para[0] == 'Theme' :
                                themes.append(para)
                            if para[0] == 'ToLoc' :
                                tolocs.append(para)

                        for theme in themes + [('Theme','None')] :
                            for toloc in tolocs + [('ToLoc','None')] :
                                if has_same_set({theme,toloc},event_para[src_key])[0] :
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
                            dst_key = theme[1]
                            range_list.append(('Theme',e_loc[dst_key]))
                            
                            for toloc in tolocs + [('ToLoc','None')] :
                                dst_key = toloc[1]
                                range_list.append(('ToLoc',e_loc[dst_key]))

                                hidden = ee.initHidden()
                                r,m = ee(bilstm_output,tr_emb,entity_emb,s_r,range_list,hidden)
                                e_flag,e_id = has_same_set({theme,toloc},event_para[src_key])
                                
                                if e_flag :
                                    if uniform(0,1) > pos_count :
                                        continue
                                    loss = loss + criterion_ee(r,pos_target)/loss_ee_resize
                                    loss_c = loss_c + criterion_ee(r,pos_target)/loss_ee_resize
                                    m_type = modification.get(e_id,'None')
                                    m_target = modification_target[m_type]
                                    loss = loss + criterion_m(m,m_target)
                                    loss_d = loss_d + criterion_m(m,m_target)
                                    freq_dict[src_type[src_key]][0] = freq_dict[src_type[src_key]][0] + 1
                                else :
                                    if uniform(0,1) > neg_count :
                                        continue
                                    loss = loss + criterion_ee(r,neg_target)/loss_ee_resize
                                    loss_c = loss_c + criterion_ee(r,neg_target)/loss_ee_resize
                                    freq_dict[src_type[src_key]][1] = freq_dict[src_type[src_key]][1] + 1

                                range_list.pop()
                            range_list.pop()

                    if src_type[src_key] == 'Infection' :
                        themes = []
                        ptps = []
                        for para in event_para_[src_key]:
                            if para[0] == 'Theme' :
                                themes.append(para)
                            if para[0] == 'Participant' :
                                ptps.append(para)

                        for theme in themes + [('Theme','None')] :
                            for ptp in ptps + [('Participant','None')] :
                                if has_same_set({theme,ptp},event_para[src_key])[0] :
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
                            dst_key = theme[1]
                            range_list.append(('Theme',e_loc[dst_key]))
                            
                            for ptp in ptps + [('Participant','None')] :
                                dst_key = ptp[1]
                                range_list.append(('Participant',e_loc[dst_key]))

                                hidden = ee.initHidden()
                                r,m = ee(bilstm_output,tr_emb,entity_emb,s_r,range_list,hidden)
                                e_flag,e_id = has_same_set({theme,ptp},event_para[src_key])
                            
                                if e_flag :
                                    if uniform(0,1) > pos_count :
                                        continue
                                    loss = loss + criterion_ee(r,pos_target)/loss_ee_resize
                                    loss_c = loss_c + criterion_ee(r,pos_target)/loss_ee_resize
                                    m_type = modification.get(e_id,'None')
                                    m_target = modification_target[m_type]
                                    loss = loss + criterion_m(m,m_target)
                                    loss_d = loss_d + criterion_m(m,m_target)
                                    freq_dict[src_type[src_key]][0] = freq_dict[src_type[src_key]][0] + 1
                                else :
                                    if uniform(0,1) > neg_count :
                                        continue
                                    loss = loss + criterion_ee(r,neg_target)/loss_ee_resize
                                    loss_c = loss_c + criterion_ee(r,neg_target)/loss_ee_resize
                                    freq_dict[src_type[src_key]][1] = freq_dict[src_type[src_key]][1] + 1

                                range_list.pop()
                            range_list.pop()

                    if src_type[src_key] == 'Gene_expression' :
                        themes = []
                        theme2s = []
                        for para in event_para_[src_key]:
                            if para[0] == 'Theme' :
                                themes.append(para)
                            if para[0] == 'Theme2' :
                                theme2s.append(para)

                        for theme in themes :
                            for theme2 in theme2s + [('Theme2','None')] :
                                if has_same_set({theme,theme2},event_para[src_key])[0] :
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
                            dst_key = theme[1]
                            range_list.append(('Theme',e_loc[dst_key]))
                            
                            for theme2 in theme2s + [('Theme2','None')] :
                                dst_key = theme2[1]
                                range_list.append(('Theme2',e_loc[dst_key]))

                                hidden = ee.initHidden()
                                r,m = ee(bilstm_output,tr_emb,entity_emb,s_r,range_list,hidden)
                                e_flag,e_id = has_same_set({theme,theme2},event_para[src_key])
                            
                                if e_flag :
                                    if uniform(0,1) > pos_count :
                                        continue
                                    loss = loss + criterion_ee(r,pos_target)/loss_ee_resize
                                    loss_c = loss_c + criterion_ee(r,pos_target)/loss_ee_resize
                                    m_type = modification.get(e_id,'None')
                                    m_target = modification_target[m_type]
                                    loss = loss + criterion_m(m,m_target)
                                    loss_d = loss_d + criterion_m(m,m_target)
                                    freq_dict[src_type[src_key]][0] = freq_dict[src_type[src_key]][0] + 1
                                else :
                                    if uniform(0,1) > neg_count :
                                        continue
                                    loss = loss + criterion_ee(r,neg_target)/loss_ee_resize
                                    loss_c = loss_c + criterion_ee(r,neg_target)/loss_ee_resize
                                    freq_dict[src_type[src_key]][1] = freq_dict[src_type[src_key]][1] + 1

                                range_list.pop()
                            range_list.pop()

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
                            for site in sites + [('Site','None')] :
                                if has_same_set({theme,site},event_para[src_key])[0] :
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
                            dst_key = theme[1]
                            range_list.append(('Theme',e_loc[dst_key]))
                            
                            for site in sites + [('Site','None')] :
                                dst_key = site[1]
                                range_list.append(('Site',e_loc[dst_key]))

                                hidden = ee.initHidden()
                                r,m = ee(bilstm_output,tr_emb,entity_emb,s_r,range_list,hidden)
                                e_flag,e_id = has_same_set({theme,site},event_para[src_key])
                            
                                if e_flag :
                                    if uniform(0,1) > pos_count :
                                        continue
                                    loss = loss + criterion_ee(r,pos_target)/loss_ee_resize
                                    loss_c = loss_c + criterion_ee(r,pos_target)/loss_ee_resize
                                    m_type = modification.get(e_id,'None')
                                    m_target = modification_target[m_type]
                                    loss = loss + criterion_m(m,m_target)
                                    loss_d = loss_d + criterion_m(m,m_target)
                                    freq_dict[src_type[src_key]][0] = freq_dict[src_type[src_key]][0] + 1
                                else :
                                    if uniform(0,1) > neg_count :
                                        continue
                                    loss = loss + criterion_ee(r,neg_target)/loss_ee_resize
                                    loss_c = loss_c + criterion_ee(r,neg_target)/loss_ee_resize
                                    freq_dict[src_type[src_key]][1] = freq_dict[src_type[src_key]][1] + 1

                                range_list.pop()
                            range_list.pop()

                    if src_type[src_key] == 'Pathway' :

                        themes = []
                        ptps = []
                        ptp2s = []
                        for para in event_para_[src_key]:
                            if para[0] == 'Theme' :
                                themes.append(para)
                            if para[0] == 'Participant' :
                                ptps.append(para)
                            if para[0] == 'Participant2' :
                                ptp2s.append(para)

                        pos_count,neg_count = 1.0,1.0
                        for theme in themes + [('Theme','None')] :
                            for ptp in ptps +[('Participant','None')]:
                                for ptp2 in ptp2s + [('Participant2','None')] :
                                    if has_same_set({theme,ptp,ptp2},event_para[src_key])[0] :
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
                            dst_key = theme[1]
                            range_list.append(('Theme',e_loc[dst_key]))
                            
                            for ptp in ptps :
                                dst_key = ptp[1]
                                range_list.append(('Participant',e_loc[dst_key]))
                                
                                for ptp2 in ptp2s + [('Participant2','None')] :
                                    dst_key = ptp2[1]
                                    range_list.append(('Participant2',e_loc[dst_key]))

                                    hidden = ee.initHidden()
                                    r,m = ee(bilstm_output,tr_emb,entity_emb,s_r,range_list,hidden)
                                    e_flag,e_id = has_same_set({theme,ptp,ptp2},event_para[src_key])
                                
                                    if e_flag :
                                        if uniform(0,1) > pos_count :
                                            continue
                                        loss = loss + criterion_ee(r,pos_target)/loss_ee_resize
                                        loss_c = loss_c + criterion_ee(r,pos_target)/loss_ee_resize
                                        m_type = modification.get(e_id,'None')
                                        m_target = modification_target[m_type]
                                        loss = loss + criterion_m(m,m_target)
                                        loss_d = loss_d + criterion_m(m,m_target)
                                        freq_dict[src_type[src_key]][0] = freq_dict[src_type[src_key]][0] + 1
                                    else :
                                        if uniform(0,1) > neg_count :
                                            continue
                                        loss = loss + criterion_ee(r,neg_target)/loss_ee_resize
                                        loss_c = loss_c + criterion_ee(r,neg_target)/loss_ee_resize
                                        freq_dict[src_type[src_key]][1] = freq_dict[src_type[src_key]][1] + 1

                                    range_list.pop()
                                range_list.pop()
                            range_list.pop()

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
                            for theme2 in theme2s + themes + [('Theme2','None')] :
                                if theme == theme2 :
                                    continue
                                for site in sites + [('Site','None')] :
                                    if has_same_set({theme,theme2,site,site},event_para[src_key])[0] :
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
                            dst_key = theme[1]
                            range_list.append(('Theme',e_loc[dst_key]))
                            
                            for theme2 in theme2s + themes + [('Theme2','None')] :
                                if theme == theme2 :
                                    continue
                                dst_key = theme2[1]
                                range_list.append(('Theme2',e_loc[dst_key]))
                            
                                for site in sites + [('Site','None')] :
                                    dst_key = site[1]
                                    range_list.append(('Site',e_loc[dst_key]))

                                    hidden = ee.initHidden()
                                    r,m = ee(bilstm_output,tr_emb,entity_emb,s_r,range_list,hidden)
                                    e_flag,e_id = has_same_set({theme,theme2,site},event_para[src_key])
                            
                                    if e_flag :
                                        if uniform(0,1) > pos_count :
                                            continue
                                        loss = loss + criterion_ee(r,pos_target)/loss_ee_resize
                                        loss_c = loss_c + criterion_ee(r,pos_target)/loss_ee_resize
                                        m_type = modification.get(e_id,'None')
                                        m_target = modification_target[m_type]
                                        loss = loss + criterion_m(m,m_target)
                                        loss_d = loss_d + criterion_m(m,m_target)
                                        freq_dict[src_type[src_key]][0] = freq_dict[src_type[src_key]][0] + 1
                                    else :
                                        if uniform(0,1) > neg_count :
                                            continue
                                        loss = loss + criterion_ee(r,neg_target)/loss_ee_resize
                                        loss_c = loss_c + criterion_ee(r,neg_target)/loss_ee_resize
                                        freq_dict[src_type[src_key]][1] = freq_dict[src_type[src_key]][1] + 1

                                    range_list.pop()
                                range_list.pop()
                            range_list.pop()

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
                            for theme2 in theme2s + [('Theme2','None')] :
                                for atloc in atlocs + [('AtLoc','None')] :
                                    for fromloc in fromlocs + [('FromLoc','None')] :
                                        for toloc in tolocs + [('ToLoc','None')] :
                                            if has_same_set({theme,theme2,atloc,fromloc,toloc},event_para[src_key])[0] :
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
                            dst_key = theme[1]
                            range_list.append(('Theme',e_loc[dst_key]))
                            
                            for theme2 in theme2s + [('Theme2','None')] :
                                dst_key = theme2[1]
                                range_list.append(('Theme2',e_loc[dst_key]))
                            
                                for atloc in atlocs + [('AtLoc','None')] :
                                    dst_key = atloc[1]
                                    range_list.append(('AtLoc',e_loc[dst_key]))
                            
                                    for fromloc in fromlocs + [('FromLoc','None')] :
                                        dst_key = fromloc[1]
                                        range_list.append(('FromLoc',e_loc[dst_key]))
                            
                                        for toloc in tolocs + [('ToLoc','None')] :
                                            dst_key = toloc[1]
                                            range_list.append(('ToLoc',e_loc[dst_key]))
                                            
                                            hidden = ee.initHidden()
                                            r,m = ee(bilstm_output,tr_emb,entity_emb,s_r,range_list,hidden)
                                            e_flag,e_id = has_same_set({theme,theme2,atloc,fromloc,toloc},event_para[src_key])
                                    
                                            if e_flag :
                                                if uniform(0,1) > pos_count :
                                                    continue
                                                loss = loss + criterion_ee(r,pos_target)/loss_ee_resize
                                                loss_c = loss_c + criterion_ee(r,pos_target)/loss_ee_resize
                                                m_type = modification.get(e_id,'None')
                                                m_target = modification_target[m_type]
                                                loss = loss + criterion_m(m,m_target)
                                                loss_d = loss_d + criterion_m(m,m_target)
                                                freq_dict[src_type[src_key]][0] = freq_dict[src_type[src_key]][0] + 1
                                            else :
                                                if uniform(0,1) > neg_count :
                                                    continue
                                                loss = loss + criterion_ee(r,neg_target)/loss_ee_resize
                                                loss_c = loss_c + criterion_ee(r,neg_target)/loss_ee_resize
                                                freq_dict[src_type[src_key]][1] = freq_dict[src_type[src_key]][1] + 1

                                            range_list.pop()
                                        range_list.pop()
                                    range_list.pop()
                                range_list.pop()
                            range_list.pop()

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
                            for cause in causes + [('Cause','None')] :
                                if has_same_set({theme,cause},event_para[src_key])[0] :
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
                            dst_key = theme[1]
                            range_list.append(('Theme',e_loc[dst_key]))
                            
                            for cause in causes + [('Cause','None')] :
                                dst_key = cause[1]
                                range_list.append(('Cause',e_loc[dst_key]))

                                hidden = ee.initHidden()
                                r,m = ee(bilstm_output,tr_emb,entity_emb,s_r,range_list,hidden)
                                e_flag,e_id = has_same_set({theme,cause},event_para[src_key])
                            
                                if e_flag :
                                    if uniform(0,1) > pos_count :
                                        continue
                                    loss = loss + criterion_ee(r,pos_target)/loss_ee_resize
                                    loss_c = loss_c + criterion_ee(r,pos_target)/loss_ee_resize
                                    m_type = modification.get(e_id,'None')
                                    m_target = modification_target[m_type]
                                    loss = loss + criterion_m(m,m_target)
                                    loss_d = loss_d + criterion_m(m,m_target)
                                    freq_dict[src_type[src_key]][0] = freq_dict[src_type[src_key]][0] + 1
                                else :
                                    if uniform(0,1) > neg_count :
                                        continue
                                    loss = loss + criterion_ee(r,neg_target)/loss_ee_resize
                                    loss_c = loss_c + criterion_ee(r,neg_target)/loss_ee_resize
                                    freq_dict[src_type[src_key]][1] = freq_dict[src_type[src_key]][1] + 1

                                range_list.pop()
                            range_list.pop()

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
                            for theme2 in theme2s + [('Theme2','None')] :
                                for instr in instrs + [('Instrument','None')] :
                                    for instr2 in instr2s + [('Instrument2','None')] :
                                        if has_same_set({theme,theme2,instr,instr2},event_para[src_key])[0] :
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
                            dst_key = theme[1]
                            range_list.append(('Theme',e_loc[dst_key]))
                            
                            for theme2 in theme2s + [('Theme2','None')] :
                                dst_key = theme2[1]
                                range_list.append(('Theme2',e_loc[dst_key]))
                            
                                for instr in instrs + [('Instrument','None')] :
                                    dst_key = instr[1]
                                    range_list.append(('Instrument',e_loc[dst_key]))
                            
                                    for instr2 in instr2s + [('Instrument2','None')] :
                                        dst_key = instr2[1]
                                        range_list.append(('Instrument2',e_loc[dst_key]))

                                        hidden = ee.initHidden()
                                        r,m = ee(bilstm_output,tr_emb,entity_emb,s_r,range_list,hidden)
                                        e_flag,e_id = has_same_set({theme,theme2,instr,instr2},event_para[src_key])
                                        
                                        if e_flag :
                                            if uniform(0,1) > pos_count :
                                                continue
                                            loss = loss + criterion_ee(r,pos_target)/loss_ee_resize
                                            loss_c = loss_c + criterion_ee(r,pos_target)/loss_ee_resize
                                            m_type = modification.get(e_id,'None')
                                            m_target = modification_target[m_type]
                                            loss = loss + criterion_m(m,m_target)
                                            loss_d = loss_d + criterion_m(m,m_target)
                                            freq_dict[src_type[src_key]][0] = freq_dict[src_type[src_key]][0] + 1
                                        else :
                                            if uniform(0,1) > neg_count :
                                                continue
                                            loss = loss + criterion_ee(r,neg_target)/loss_ee_resize
                                            loss_c = loss_c + criterion_ee(r,neg_target)/loss_ee_resize
                                            freq_dict[src_type[src_key]][1] = freq_dict[src_type[src_key]][1] + 1

                                        range_list.pop()
                                    range_list.pop()
                                range_list.pop()
                            range_list.pop()
                               
	    loss.backward()
	    optimizer.step()

            running_loss += loss.data[0]
            if i%10 == 9:
                print('[%d, %3d] loss: %.5f %.5f %.5f %.5f %.5f' % (epoch+1,i+1,running_loss/160,loss_a/160,loss_b/160,loss_c/160,loss_d/160)) # step is 10 and batch_size is 8   
                running_loss = 0.0
                loss_a,loss_b,loss_c,loss_d = 0.0, 0.0, 0.0, 0.0

        if (epoch+1) % 10 == 0:
            for key in freq_dict :
                if freq_dict[key][0]+freq_dict[key][1] == 0 :
                    continue
                print key,'%d %d %.5f'%(freq_dict[key][0],freq_dict[key][1],float(freq_dict[key][0])/(freq_dict[key][0]+freq_dict[key][1]))
                
            torch.save(char_cnn.state_dict(),path_+'/nets/char_cnn_%d_%d.pth'%(epoch+1,number))
            torch.save(bilstm.state_dict(),path_+'/nets/bilstm_%d_%d.pth'%(epoch+1,number))
            torch.save(tr.state_dict(),path_+'/nets/tr_%d_%d.pth'%(epoch+1,number))
            torch.save(rc.state_dict(),path_+'/nets/rc_%d_%d.pth'%(epoch+1,number))
            torch.save(ee.state_dict(),path_+'/nets/ee_%d_%d.pth'%(epoch+1,number))

    print('Finished Training')
