import os
import sys
import math
import torch
import random
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
from define_net_pretrain import Char_CNN_pretrain
from define_net import Char_CNN_encode,BiLSTM,Trigger_Recognition,Relation_Classification
from define_net_event_evaluation import *
from entity_event_dict_CG import *
from generate_utils import *
from word_set import event_to_onehot
from sklearn.metrics import roc_auc_score as auc


def load_pretrain_vector(word_index,word_embedding):

    filename = os.path.abspath('.') + '/network/myword2vec'
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

    seta = set(seta)
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
    if len(subsetb.keys()) == 0:
        subsetb['E_empty'] = set()

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

    trainset = Sentence_Set(path_+'/table/',new_dict=False) # be True at the first running before the source changed
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
    trainloader = DataLoader(trainset,batch_size=16,shuffle=True)
    testloader = DataLoader(testset,batch_size=16,shuffle=True)

    char_cnn = Char_CNN_encode(char_dim).cuda()
    bilstm = BiLSTM(word_dim,entity_dim).cuda()
    tr = Trigger_Recognition(event_dim).cuda()
    rc = Relation_Classification(relation_dim).cuda()
    ee = Event_Evaluation().cuda()

    ccp = Char_CNN_pretrain(char_dim,event_dim)
    ccp.load_state_dict(torch.load(path_+'/network/char_rnn_pretrain.pth'))
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
    init.xavier_uniform(rc.position_embedding.weight)
    init.xavier_uniform(rc.empty_embedding.weight)
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

    f = file(path_+'/word_index', 'r')
    word_index = cPickle.load(f)

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

    word_embedding = bilstm.state_dict()['word_embedding.weight'].cpu().numpy()
    word_embedding = load_pretrain_vector(word_index,word_embedding)
    pretrained_weight = np.array(word_embedding)
    bilstm.word_embedding.weight.data.copy_(torch.from_numpy(pretrained_weight).cuda())
    '''     
    for p in bilstm.fixed_embedding.parameters():
	    p.requires_grad = False
    '''
    class_weight_tr = [ 5 for i in range(0,event_dim) ] #
    class_weight_tr[event_index['NONE']] = 1
    class_weight_tr = torch.FloatTensor(class_weight_tr).cuda()
    criterion_tr = nn.NLLLoss(weight=class_weight_tr)

    class_weight_rc = [ 5 for i in range(0,relation_dim) ] #
    class_weight_rc[relation_index['NONE']] = 1
    class_weight_rc = torch.FloatTensor(class_weight_rc).cuda()
    criterion_rc = nn.NLLLoss(weight=class_weight_rc)

    class_weight_ee = [1,1] #
    class_weight_ee = torch.FloatTensor(class_weight_ee).cuda()
    criterion_ee = nn.NLLLoss(weight=class_weight_ee)

    class_weight_m = [1,5,5] #
    class_weight_m = torch.FloatTensor(class_weight_m).cuda()
    criterion_m = nn.NLLLoss(weight=class_weight_m)

    optimizer = optim.Adam(list(char_cnn.parameters())+ \
                           list(bilstm.parameters())+ \
                           list(tr.parameters())+ \
                           list(rc.parameters())+ \
                           list(ee.parameters()), lr=0.007, weight_decay=0.0002)

    pos_target = Variable(torch.LongTensor([1])).cuda()
    neg_target = Variable(torch.LongTensor([0])).cuda()

    modification_target = dict()
    modification_target['None'] = Variable(torch.LongTensor([0])).cuda()
    modification_target['Speculation'] = Variable(torch.LongTensor([1])).cuda()
    modification_target['Negation'] = Variable(torch.LongTensor([2])).cuda()

    for epoch in range(100): #

        running_loss = 0.0
        loss_a,loss_b,loss_c,loss_d = 0.0,0.0,0.0,0.0
        addition_count = 0.0
        tr_validation, rc_validation, ee_validation = [],[],[]
        sigma = 10.0/(10.0+math.exp(epoch/10.0))

        for i,batch in enumerate(trainloader,0):

            loss = 0
            optimizer.zero_grad()

            for data in batch: # due to we have modified the defination of batch, the batch here is a list

                input_word, input_entity, input_char, target, entity_loc, event_loc, relation, event_para, modification, fname = data
                input_word, input_entity, target = Variable(input_word).cuda(),Variable(input_entity).cuda(),Variable(target.view(-1)).cuda() # L of indices

                char_encode = []
                for chars in input_char :
                    chars = char_cnn(Variable(chars).cuda())
                    char_encode.append(chars)
                char_encode = torch.cat(char_encode,0) # L*N*conv_out_channel

                hidden = bilstm.initHidden()
                bilstm_output,hidden,entity_emb = bilstm((input_word,input_entity,char_encode),hidden)

                if random.uniform(0,1) < sigma :
                    infer = False
                else :
                    infer = True
                tr_output,tr_emb,tr_index = tr(bilstm_output,target,infer)

                loss = loss + criterion_tr(tr_output,target) # NLLloss only accept 1-dimension
                loss_a = loss_a + criterion_tr(tr_output,target)

                for j in range(0,target.size()[0]):
                    if event_index_r[int(target[j])] != 'NONE' :
                        tmp_flag = 1
                    else :
                        tmp_flag = 0
                    tmp_score = float(tr_output[j][target[j]] - tr_output[j][event_index['NONE']])
                    tr_validation.append((tmp_flag, tmp_score))
                    if int(target[j]) != int(tr_index[j]) :
                        vector = tr.get_event_embedding(int(target[j]))
                        tr_emb[j] = vector

                e_loc = dict(entity_loc.items()+event_loc.items())

                event_para_ = dict()
                src_type = dict()

                for rlt in relation.keys() :

                    src_key = rlt[0]
                    dst_key = rlt[1]
                    rlt_target = Variable(relation[rlt]).cuda()

                    src_begin,src_end = e_loc[src_key]
                    dst_begin,dst_end = e_loc[dst_key]
                    '''
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
                    src_name = event_index_r[int(target[src_begin])].split('-')[0]
                    src_type[src_key] = src_name

                    src = torch.cat((bilstm_output[src_begin:src_end+1],\
                                     tr_emb[src_begin:src_end+1],entity_emb[src_begin:src_end+1]),2) #,entity_emb[src_begin:src_end+1]

                    if event_loc.has_key(dst_key):
                        event_flag = True
                        dst = torch.cat((bilstm_output[dst_begin:dst_end+1], \
                                         tr_emb[dst_begin:dst_end+1],entity_emb[dst_begin:dst_end+1]),2) # ,entity_emb[dst_begin:dst_end+1]
                    else :
                        event_flag = False
                        dst = torch.cat((bilstm_output[dst_begin:dst_end+1], \
                                         tr_emb[dst_begin:dst_end+1],entity_emb[dst_begin:dst_end+1]),2) # tr_emb[dst_begin:dst_end+1],

                    if event_flag and not src_name in {'Regulation','Positive_regulation','Negative_regulation','Planned_process'} :
                        continue

                    reverse_flag = False
                    if src_begin > dst_begin :
                        reverse_flag = True

                    #middle_flag = False
                    if  src_end+1 < dst_begin:
                        middle = torch.cat((bilstm_output[src_end+1:dst_begin], \
                                            tr_emb[src_end+1:dst_begin],entity_emb[src_end+1:dst_begin]),2)
                    elif dst_end < src_begin-1:
                        middle = torch.cat((bilstm_output[dst_end+1:src_begin], \
                                            tr_emb[dst_end+1:src_begin],entity_emb[dst_end+1:src_begin]),2)
                    else : # adjacent or overlapped
                        #middle_flag = True
                        middle = torch.FloatTensor(0)

                    rc_output = rc(src,middle,dst,reverse_flag,event_flag)

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
                        tmp_flag = 1
                        if src_key not in event_para.keys() :
                            continue # detail in notebook
                        if not event_para_.has_key(src_key) :
                            event_para_[src_key] = set()
                            event_para_[src_key].add((current_type,dst_key))
                        else :
                            event_para_[src_key].add((current_type,dst_key))
                    else :
                        tmp_flag = 0
                    tmp_score = float(rc_output[0][int(correct_index)] - rc_output[0][relation_index['NONE']])
                    rc_validation.append((tmp_flag, tmp_score))

                e_loc['None'] = (0,-1)
                entity_loc['None'] = (0,-1)
                event_loc['None'] = (0,-1)

                for src_key in event_para.keys():
                    if not event_para_.has_key(src_key) :
                        event_para_[src_key] = set()
                    if not src_type.has_key(src_key) :
                        src_begin = event_loc[src_key][0]
                        src_type[src_key] = event_index_r[int(target[src_begin])].split('-')[0]

                loss_ee_resize = 2.0
                
                for src_key in event_para_.keys() :

                    s_r = e_loc[src_key]
                    pos_count,neg_count = 1.0,1.0

                    range_lists,para_sets = generate_all_combination(src_type[src_key],event_para_[src_key],e_loc)

                    for j in range(0,len(range_lists)):
                        range_list = range_lists[j]
                        para_set = para_sets[j]

                        if has_same_set(para_set,event_para[src_key])[0] :
                            pos_count = pos_count + 1
                        else :
                            neg_count = neg_count + 1
                    '''
		    # another class-balance method by generating negative samples in ratio=1:1 with positive samples
                    while neg_count < pos_count :
                        j = random.randint(0,len(para_sets)-1)
                        para_set = para_sets[j][:]
                        range_list = range_lists[j][:]

                        random_para_index = random.randint(0,len(para_set)-1)
                        replaced_type,replaced_para = para_set[random_para_index]
                        if replaced_para in entity_loc.keys() :
                            tmp_random = random.randint(0,len(entity_loc.keys())-1)
                            corrupt_para = entity_loc.keys()[tmp_random]
                        elif replaced_para in event_loc.keys() :
                            tmp_random = random.randint(0,len(event_loc.keys())-1)
                            corrupt_para = event_loc.keys()[tmp_random]

                        para_set[random_para_index] = (replaced_type,corrupt_para)
                        range_list[random_para_index] = (replaced_type,e_loc[corrupt_para])
                        para_sets.append(para_set)
                        range_lists.append(range_list)
                        neg_count = neg_count + 1
                    '''
                    if pos_count < neg_count :
                        min_count = pos_count
                    else :
                        min_count = neg_count
                    pos_count = min_count/pos_count
                    neg_count = min_count/neg_count

                    for j in range(0,len(range_lists)):
                        range_list = range_lists[j]
                        para_set = para_sets[j]
                        '''
			# another class-balance method by generating negative samples in ratio=1:1 with positive samples
                        e_flag,e_id = has_same_set(para_set,event_para[src_key])
                        if e_flag :
                            hidden = ee.initHidden()
                            r,m = ee(bilstm_output,tr_emb,entity_emb,s_r,range_list,hidden)
                            loss = loss + criterion_ee(r,pos_target)/loss_ee_resize
                            loss_c = loss_c + criterion_ee(r,pos_target)/loss_ee_resize
                            m_type = modification.get(e_id,'None')
                            m_target = modification_target[m_type]
                            loss = loss + criterion_m(m,m_target)
                            loss_d = loss_d + criterion_m(m,m_target)

                            tmp_score = float(r[0][1] - r[0][0])
                            ee_validation.append((1, tmp_score))

                            random_para_index = random.randint(0,len(para_set)-1)
                            replaced_type,replaced_para = para_set[random_para_index]
                            if replaced_para in entity_loc.keys() :
                                tmp_random = random.randint(0,len(entity_loc.keys())-1)
                                corrupt_para = entity_loc.keys()[tmp_random]
                            elif replaced_para in event_loc.keys() :
                                tmp_random = random.randint(0,len(event_loc.keys())-1)
                                corrupt_para = event_loc.keys()[tmp_random]
                            else :
                                tmp_random = random.randint(0,len(e_loc.keys())-1)
                                corrupt_para = e_loc.keys()[tmp_random]
                            range_list[random_para_index] = (replaced_type,e_loc[corrupt_para])

                            hidden = ee.initHidden()
                            r_n,m_n = ee(bilstm_output,tr_emb,entity_emb,s_r,range_list,hidden)
                            loss = loss + criterion_ee(r_n,neg_target)/loss_ee_resize
                            loss_c = loss_c + criterion_ee(r_n,neg_target)/loss_ee_resize

                            tmp_score = float(r_n[0][1] - r_n[0][0])
                            ee_validation.append((0, tmp_score))
                        '''
                        hidden = ee.initHidden()
                        r,m = ee(bilstm_output,tr_emb,entity_emb,s_r,range_list,hidden)
                        e_flag,e_id = has_same_set(para_set,event_para[src_key])

                        if e_flag :
                            if uniform(0,1) > pos_count :
                                continue
                            loss = loss + criterion_ee(r,pos_target)/loss_ee_resize
                            loss_c = loss_c + criterion_ee(r,pos_target)/loss_ee_resize
                            m_type = modification.get(e_id,'None')
                            m_target = modification_target[m_type]
                            loss = loss + criterion_m(m,m_target)
                            loss_d = loss_d + criterion_m(m,m_target)
                            tmp_flag = 1
                        else :
                            if uniform(0,1) > neg_count :
                                continue
                            loss = loss + criterion_ee(r,neg_target)/loss_ee_resize
                            loss_c = loss_c + criterion_ee(r,neg_target)/loss_ee_resize
                            tmp_flag = 0
                        
                        tmp_score = float(r[0][1] - r[0][0])
                        ee_validation.append((tmp_flag, tmp_score))

            loss.backward()
            optimizer.step()
            #print tr_loss, rc_loss, ee_loss, m_loss
            running_loss += loss.data[0]
            if i%10 == 9:
                print('[%d, %3d] loss: %.5f %.5f %.5f %.5f %.5f' % (epoch+1,i+1,running_loss/160,loss_a/160,loss_b/160,loss_c/160,loss_d/160)) # step is 10 and batch_size is 8
                running_loss = 0.0
                loss_a,loss_b,loss_c,loss_d = 0.0, 0.0, 0.0, 0.0

        tr_validation_sorted = sorted(tr_validation, key=lambda x:x[0], reverse=True)
        tr_auc_score = auc([ r[0] for r in tr_validation_sorted ], [ r[1] for r in tr_validation_sorted ])
        rc_validation_sorted = sorted(rc_validation, key=lambda x:x[0], reverse=True)
        rc_auc_score = auc([ r[0] for r in rc_validation_sorted ], [ r[1] for r in rc_validation_sorted ])
        ee_validation_sorted = sorted(ee_validation, key=lambda x:x[0], reverse=True)
        ee_auc_score = auc([ r[0] for r in ee_validation_sorted ], [ r[1] for r in ee_validation_sorted ])
        print tr_auc_score, rc_auc_score, ee_auc_score

        if (epoch+1) % 10 == 0:
            torch.save(char_cnn.state_dict(),path_+'/network/char_cnn_%d_%d.pth'%(epoch+1,number))
            torch.save(bilstm.state_dict(),path_+'/network/bilstm_%d_%d.pth'%(epoch+1,number))
            torch.save(tr.state_dict(),path_+'/network/tr_%d_%d.pth'%(epoch+1,number))
            torch.save(rc.state_dict(),path_+'/network/rc_%d_%d.pth'%(epoch+1,number))
            torch.save(ee.state_dict(),path_+'/network/ee_%d_%d.pth'%(epoch+1,number))

    print('Finished Training')
