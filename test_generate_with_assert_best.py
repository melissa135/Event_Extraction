import os
import csv
import torch
import cPickle
import torch.nn.init as init
from torch.autograd import Variable
from pandas.io.parsers import read_csv
from dataloader_modified import DataLoader
from define_net import Char_CNN_encode,BiLSTM,NER,RC
from sentence_set_single import Sentence_Set_Single
from define_net_event_assert import *
from entity_event_dict import *


class Event :
    def __init__(self, event_id, paras, support):
        self.event_id = event_id
        self.paras = paras
        self.support = support
        self.choosed = False

    def get_all_para_nodes(self):
        para_nodes = set()
        for para in self.paras :
            para_nodes.add(para[1])
        return para_nodes

    def set_choosed(self):
        self.choosed = True


def remove_subset(valid_combination):
  
    valid_combination_without_none = [] # remove none-parameter
    for v_c in valid_combination :
        v_c_without_none = set()
        for para in v_c.paras :
            if not para[1] == 'None' :
                v_c_without_none.add(para)
        event = Event(v_c.event_id,v_c_without_none,v_c.support)
        valid_combination_without_none.append(event)
                
    valid_combination = []
    for v_c in valid_combination_without_none :
        subset_flag = False
        for v_c_ in valid_combination_without_none :
            if v_c == v_c_ :
                continue
            if v_c.paras.issubset(v_c_.paras) or v_c_.paras.issubset(v_c.paras) :
                if v_c.support < v_c_.support : # delete the small-support one
                    subset_flag = True
        if not subset_flag :
            valid_combination.append(v_c) # remove subset
    
    return valid_combination


if __name__ == '__main__':

    path_ = os.path.abspath('.')
    folder = '/home/zhu/event_extraction/table_test/'

    first_load = True
    
    for root, _, fnames in os.walk(folder):
        for fname in fnames:
            print fname
            
            testset = Sentence_Set_Single(folder+fname)
            testloader = DataLoader(testset,batch_size=1,shuffle=False)
            df = read_csv(folder+fname)
            #df['word'] = df['word'].astype('string') # if not, it will mistake 'null' as nan

            char_dim = testset.get_char_dim()
            word_dim = testset.get_word_dim()
            entity_dim = testset.get_entity_dim()
            event_dim = testset.get_event_dim()
            relation_dim = testset.get_relation_dim()

            if first_load :

                char_cnn = Char_CNN_encode(char_dim)
                bilstm = BiLSTM(word_dim,entity_dim)
                ner = NER(event_dim)
                rc = RC(relation_dim)

                smp = Simple_Max_Pooling()
                none_embedding = Simple_None_Embedding()
                event_assert = Event_Assert(41,12,59)
           
                epoch = 20
                number = 4
                char_cnn.load_state_dict(torch.load(path_+'/char_cnn_%d_%d.pth'%(epoch,number)))
                bilstm.load_state_dict(torch.load(path_+'/bilstm_%d_%d.pth'%(epoch,number)))
                ner.load_state_dict(torch.load(path_+'/ner_%d_%d.pth'%(epoch,number)))
                rc.load_state_dict(torch.load(path_+'/rc_%d_%d.pth'%(epoch,number)))
                none_embedding.load_state_dict(torch.load(path_+'/none_embedding_%d_%d.pth'%(epoch,number)))
                event_assert.load_state_dict(torch.load(path_+'/assert_nets/event_assert_%d_%d.pth'%(epoch,number)))
               
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

                first_load = False

            t_index = 1000 # Entity number is unimportant, begin with a large number
            e_index = 1
            base_loc = 0

            result_dir = folder+fname
            result_dir = result_dir.replace('table_test','a2_result')
            result_dir = result_dir.replace('.csv','.a2')
            f = file(result_dir,'w')
            
            for i,batch in enumerate(testloader,0):
                    
                for data in batch: # due to we have modified the defination of batch, the batch here is a list

                    input_word, input_entity, input_char, target, entity_loc, event_loc, relation, event_para = data
                    input_word, input_entity, target = Variable(input_word),Variable(input_entity),Variable(target)
                
                    char_encode = []
                    for chars in input_char :
                        chars = char_cnn(Variable(chars))
                        char_encode.append(chars)
                    char_encode = torch.cat(char_encode,0) # L*N*conv_out_channel

                    hidden1,hidden2 = bilstm.initHidden()
                    bilstm.eval()
                    bilstm_output,hidden,entity_emb = bilstm((input_word,input_entity,char_encode),(hidden1,hidden2))
                    ner.eval()
                    ner_output,ner_emb,ner_index = ner(bilstm_output)
                 
                    row = ner_index
                    previous_type = 'NONE'

                    event_loc_ = dict()
                    event_info = dict()
                    event_argu = dict()
                    
                    for k in range(0,len(row)) :
                        
                        index = row[k]
                        current_type = event_index_r[index].split('-')[0]
                        if current_type != previous_type :
                            if k != 0 and previous_type != 'NONE' : # the first one has no previous recongition
                                t_notation = 'T%d'%t_index
                                t_index = t_index + 1
                                event_info[t_notation] = ( previous_type,start_postion,end_postion,words )
                                event_loc_[t_notation] = ( start_loc,end_loc )
                            if current_type != 'NONE' : # begin a new event
                                start_loc,end_loc = k,k
                                start_postion = df['position'][k+base_loc]
                                try :
                                    end_postion = df['position'][k+base_loc]+len(df['word'][k+base_loc])
                                    words = df['word'][k+base_loc]
                                except Exception as e :
                                    print df['word'][k+base_loc]
                                    end_postion = df['position'][k+base_loc]+len(str(df['word'][k+base_loc])) + 1
                                    words = 'null'
                        else :
                            if current_type != 'NONE' :
                                end_loc = k
                                try :
                                    end_postion = df['position'][k+base_loc]+len(df['word'][k+base_loc])
                                    words = df['word'][k+base_loc]
                                except Exception as e :
                                    print df['word'][k+base_loc]
                                    end_postion = df['position'][k+base_loc]+len(str(df['word'][k+base_loc])) + 1
                                    words = 'null'
                        previous_type = current_type

                    if index != 0 :
                        t_notation = 'T%d'%t_index
                        t_index = t_index + 1
                        event_info[t_notation] = ( previous_type,start_postion,end_postion,words )
                        event_loc_[t_notation] = ( start_loc,end_loc )
                    
                    event_support = dict()
                    for key in event_loc_.keys() :
                        support = 0.0
                        start,end = event_loc_[key]
                        for j in range(start,end+1) :
                            index = row[j]
                            support = support + float(ner_output[j][index])-float(ner_output[j][0])
                        support = support/(end-start+1)
                        event_support[key] = support
                    
                    base_loc = base_loc+len(row)
                    
                    e_loc = dict(entity_loc.items()+event_loc_.items()) # event refer to net output
                    
                    event_para_ = dict()
                    rc_hidden_dict = dict()
                    src_type = dict()
                    dst_type = dict()
                    relation_support = dict()
                    
                    for src_key in event_loc_.keys() :
                       
                        for dst_key in e_loc.keys() :
                            if src_key == dst_key :
                                continue
                            src_range = event_loc_[src_key]
                            dst_range = e_loc[dst_key]
                            
                            src_begin,src_end = event_loc_[src_key]
                            dst_begin,dst_end = e_loc[dst_key]
                            src = bilstm_output[src_begin:src_end+1]
                            src_event = ner_emb[src_begin:src_end+1]
                            dst = bilstm_output[dst_begin:dst_end+1]

                            src_name = event_index_r[ner_index[src_begin]].split('-')[0]
                            src_type[src_key] = src_name
                            
                            event_flag = False
                            if event_loc_.has_key(dst_key):
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

                            rc.eval()
                            rc_output,rc_hidden = rc(src_event,src,middle,dst,dst_event,reverse_flag)
                            
                            row = rc_output.data
                            this_row = list(row[0])
                            index = this_row.index(max(this_row))
                            current_type = relation_index_r[index]
                            current_strength = this_row[index]

                            if not event_para_.has_key(src_key) : # key point, keep the event with no relations
                                event_para_[src_key] = set()
                            
                            if current_type != 'NONE' :
                                dst = smp(dst)
                                rc_hidden_dict[(src_key,dst_key)] = (dst,current_strength)#rc_output#Variable(row[:])
                                support = this_row[index] - this_row[2]
                                relation_support[(src_key,dst_key)] = support 
                                event_para_[src_key].add((current_type,dst_key))
                    
                    dst_type['None'] = 'None'
                    
                    default_prob = (none_embedding(0),-5.12)
                    event_info_paras = dict()
                    t2e = dict()
                    
                    for src_key in event_para_.keys() :
                        src_ = torch.LongTensor(1,1).zero_()
                        src_[0][0] = event_dict[src_type[src_key]]
                        src_ = Variable(src_)

                        candidate_event = []
                        
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
                                rlt_1 = torch.LongTensor(1,1).zero_()
                                rlt_1[0][0] = relation_index['Theme']
                                rlt_1 = Variable(rlt_1)
                                dst_key_1 = theme[1]
                                dst_1 = torch.LongTensor(1,1).zero_()
                                dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                                dst_1 = Variable(dst_1)
                                
                                r = event_assert(src_,[(rlt_1,dst_1,rc_hidden_dict.get((src_key,dst_key_1),default_prob))] )
                                support = float(r.data[0][1] - r.data[0][0])
                                event = Event(src_key,{('Theme',dst_key_1)},support)
                                
                                candidate_event.append(event)
 
                        if src_type[src_key] == 'Cell_death' or \
                           src_type[src_key] == 'Amino_acid_catabolism' or \
                           src_type[src_key] == 'Glycolysis':

                            themes = []
                            for para in event_para_[src_key]:
                                if para[0] == 'Theme' :
                                    themes.append(para)
                            
                            for theme in themes + [('Theme','None')] :
                                rlt_1 = torch.LongTensor(1,1).zero_()
                                rlt_1[0][0] = relation_index['Theme']
                                rlt_1 = Variable(rlt_1)
                                dst_key_1 = theme[1]
                                dst_1 = torch.LongTensor(1,1).zero_()
                                dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                                dst_1 = Variable(dst_1)
                                
                                r = event_assert(src_,[(rlt_1,dst_1,rc_hidden_dict.get((src_key,dst_key_1),default_prob))] )
                                support = float(r.data[0][1] - r.data[0][0])
                                event = Event(src_key,{('Theme',dst_key_1)},support)
                                
                                candidate_event.append(event)
                           
                        if src_type[src_key] == 'Cell_differentiation' or \
                           src_type[src_key] == 'Cell_transformation':
                            
                            themes = []
                            atlocs = []
                            for para in event_para_[src_key]:
                                if para[0] == 'Theme' :
                                    themes.append(para)
                                if para[0] == 'AtLoc' :
                                    atlocs.append(para)
                            
                            for theme in themes :
                                rlt_1 = torch.LongTensor(1,1).zero_()
                                rlt_1[0][0] = relation_index['Theme']
                                rlt_1 = Variable(rlt_1)
                                dst_key_1 = theme[1]
                                dst_1 = torch.LongTensor(1,1).zero_()
                                dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                                dst_1 = Variable(dst_1)
                                
                                for atloc in atlocs + [('AtLoc','None')] :
                                    rlt_2 = torch.LongTensor(1,1).zero_()
                                    rlt_2[0][0] = relation_index['AtLoc']
                                    rlt_2 = Variable(rlt_2)
                                    dst_key_2 = atloc[1]
                                    dst_2 = torch.LongTensor(1,1).zero_()
                                    dst_2[0][0] = e_dict[dst_type[dst_key_2]]
                                    dst_2 = Variable(dst_2)
                                    
                                    r = event_assert(src_,[(rlt_1,dst_1,rc_hidden_dict.get((src_key,dst_key_1),default_prob)),\
                                                           (rlt_2,dst_2,rc_hidden_dict.get((src_key,dst_key_2),default_prob))] )
                                    support = float(r.data[0][1] - r.data[0][0])
                                    event = Event(src_key,{('Theme',dst_key_1),('AtLoc',dst_key_2)},support)
                                    
                                    candidate_event.append(event)
                            
                        if src_type[src_key] == 'Blood_vessel_development' or \
                           src_type[src_key] == 'Carcinogenesis' :
                            
                            themes = []
                            atlocs = []
                            for para in event_para_[src_key]:
                                if para[0] == 'Theme' :
                                    themes.append(para)
                                if para[0] == 'AtLoc' :
                                    atlocs.append(para)
                          
                            for theme in themes + [('Theme','None')] :
                                rlt_1 = torch.LongTensor(1,1).zero_()
                                rlt_1[0][0] = relation_index['Theme']
                                rlt_1 = Variable(rlt_1)
                                dst_key_1 = theme[1]
                                dst_1 = torch.LongTensor(1,1).zero_()
                                dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                                dst_1 = Variable(dst_1)
                                
                                for atloc in atlocs + [('AtLoc','None')] :
                                    rlt_2 = torch.LongTensor(1,1).zero_()
                                    rlt_2[0][0] = relation_index['AtLoc']
                                    rlt_2 = Variable(rlt_2)
                                    dst_key_2 = atloc[1]
                                    dst_2 = torch.LongTensor(1,1).zero_()
                                    dst_2[0][0] = e_dict[dst_type[dst_key_2]]
                                    dst_2 = Variable(dst_2)
                                   
                                    r = event_assert(src_,[(rlt_1,dst_1,rc_hidden_dict.get((src_key,dst_key_1),default_prob)),\
                                                           (rlt_2,dst_2,rc_hidden_dict.get((src_key,dst_key_2),default_prob))] )
                                    support = float(r.data[0][1] - r.data[0][0])
                                    event = Event(src_key,{('Theme',dst_key_1),('AtLoc',dst_key_2)},support)
                                   
                                    candidate_event.append(event)
                            
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
                           
                            for theme in themes :
                                rlt_1 = torch.LongTensor(1,1).zero_()
                                rlt_1[0][0] = relation_index['Theme']
                                rlt_1 = Variable(rlt_1)
                                dst_key_1 = theme[1]
                                dst_1 = torch.LongTensor(1,1).zero_()
                                dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                                dst_1 = Variable(dst_1)
                                
                                for atloc in atlocs + [('AtLoc','None')] :
                                    rlt_2 = torch.LongTensor(1,1).zero_()
                                    rlt_2[0][0] = relation_index['AtLoc']
                                    rlt_2 = Variable(rlt_2)
                                    dst_key_2 = atloc[1]
                                    dst_2 = torch.LongTensor(1,1).zero_()
                                    dst_2[0][0] = e_dict[dst_type[dst_key_2]]
                                    dst_2 = Variable(dst_2)
                                    
                                    for site in sites + [('Site','None')] :
                                        rlt_3 = torch.LongTensor(1,1).zero_()
                                        rlt_3[0][0] = relation_index['Site']
                                        rlt_3 = Variable(rlt_3)
                                        dst_key_3 = site[1]
                                        dst_3 = torch.LongTensor(1,1).zero_()
                                        dst_3[0][0] = e_dict[dst_type[dst_key_3]]
                                        dst_3 = Variable(dst_3)
                                        
                                        r = event_assert(src_,[(rlt_1,dst_1,rc_hidden_dict.get((src_key,dst_key_1),default_prob)),\
                                                               (rlt_2,dst_2,rc_hidden_dict.get((src_key,dst_key_2),default_prob)),\
                                                               (rlt_3,dst_3,rc_hidden_dict.get((src_key,dst_key_3),default_prob))] )
                                        support = float(r.data[0][1] - r.data[0][0])
                                        event = Event(src_key,{('Theme',dst_key_1),('AtLoc',dst_key_2),('Site',dst_key_3)},support)
                                        
                                        candidate_event.append(event)
                            
                        if src_type[src_key] == 'Metastasis' :
                            
                            themes = []
                            tolocs = []
                            for para in event_para_[src_key]:
                                if para[0] == 'Theme' :
                                    themes.append(para)
                                if para[0] == 'ToLoc' :
                                    tolocs.append(para)
                            
                            for theme in themes + [('Theme','None')] :
                                rlt_1 = torch.LongTensor(1,1).zero_()
                                rlt_1[0][0] = relation_index['Theme']
                                rlt_1 = Variable(rlt_1)
                                dst_key_1 = theme[1]
                                dst_1 = torch.LongTensor(1,1).zero_()
                                dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                                dst_1 = Variable(dst_1)
                                
                                for toloc in tolocs :
                                    rlt_2 = torch.LongTensor(1,1).zero_()
                                    rlt_2[0][0] = relation_index['ToLoc']
                                    rlt_2 = Variable(rlt_2)
                                    dst_key_2 = toloc[1]
                                    dst_2 = torch.LongTensor(1,1).zero_()
                                    dst_2[0][0] = e_dict[dst_type[dst_key_2]]
                                    dst_2 = Variable(dst_2)
                                    
                                    r = event_assert(src_,[(rlt_1,dst_1,rc_hidden_dict.get((src_key,dst_key_1),default_prob)),\
                                                           (rlt_2,dst_2,rc_hidden_dict.get((src_key,dst_key_2),default_prob))] )
                                    support = float(r.data[0][1] - r.data[0][0])
                                    event = Event(src_key,{('Theme',dst_key_1),('ToLoc',dst_key_2)},support)
                                    
                                    candidate_event.append(event)

                        if src_type[src_key] == 'Infection' :
                            
                            themes = []
                            ptps = []
                            for para in event_para_[src_key]:
                                if para[0] == 'Theme' :
                                    themes.append(para)
                                if para[0] == 'Participant' :
                                    ptps.append(para)
                            
                            for theme in themes + [('Theme','None')] :
                                rlt_1 = torch.LongTensor(1,1).zero_()
                                rlt_1[0][0] = relation_index['Theme']
                                rlt_1 = Variable(rlt_1)
                                dst_key_1 = theme[1]
                                dst_1 = torch.LongTensor(1,1).zero_()
                                dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                                dst_1 = Variable(dst_1)
                                
                                for ptp in ptps + [('Participant','None')] :
                                    rlt_2 = torch.LongTensor(1,1).zero_()
                                    rlt_2[0][0] = relation_index['Participant']
                                    rlt_2 = Variable(rlt_2)
                                    dst_key_2 = ptp[1]
                                    dst_2 = torch.LongTensor(1,1).zero_()
                                    dst_2[0][0] = e_dict[dst_type[dst_key_2]]
                                    dst_2 = Variable(dst_2)
                                  
                                    r = event_assert(src_,[(rlt_1,dst_1,rc_hidden_dict.get((src_key,dst_key_1),default_prob)),\
                                                           (rlt_2,dst_2,rc_hidden_dict.get((src_key,dst_key_2),default_prob))] )
                                    support = float(r.data[0][1] - r.data[0][0])
                                    event = Event(src_key,{('Theme',dst_key_1),('Participant',dst_key_2)},support)
                                    
                                    candidate_event.append(event)

                        if src_type[src_key] == 'Gene_expression' :
                            
                            themes = []
                            theme2s = []
                            for para in event_para_[src_key]:
                                if para[0] == 'Theme' :
                                    themes.append(para)
                                if para[0] == 'Theme2' :
                                    theme2s.append(para)
                            
                            for theme in themes :
                                rlt_1 = torch.LongTensor(1,1).zero_()
                                rlt_1[0][0] = relation_index['Theme']
                                rlt_1 = Variable(rlt_1)
                                dst_key_1 = theme[1]
                                dst_1 = torch.LongTensor(1,1).zero_()
                                dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                                dst_1 = Variable(dst_1)
                                
                                for theme2 in theme2s + [('Theme2','None')] :
                                    rlt_2 = torch.LongTensor(1,1).zero_()
                                    rlt_2[0][0] = relation_index['Theme2']
                                    rlt_2 = Variable(rlt_2)
                                    dst_key_2 = theme2[1]
                                    dst_2 = torch.LongTensor(1,1).zero_()
                                    dst_2[0][0] = e_dict[dst_type[dst_key_2]]
                                    dst_2 = Variable(dst_2)
                                    
                                    r = event_assert(src_,[(rlt_1,dst_1,rc_hidden_dict.get((src_key,dst_key_1),default_prob)),\
                                                           (rlt_2,dst_2,rc_hidden_dict.get((src_key,dst_key_2),default_prob))] )
                                    support = float(r.data[0][1] - r.data[0][0])
                                    event = Event(src_key,{('Theme',dst_key_1),('Theme2',dst_key_2)},support)
                                    
                                    candidate_event.append(event)
 
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
                            
                            for theme in themes :
                                rlt_1 = torch.LongTensor(1,1).zero_()
                                rlt_1[0][0] = relation_index['Theme']
                                rlt_1 = Variable(rlt_1)
                                dst_key_1 = theme[1]
                                dst_1 = torch.LongTensor(1,1).zero_()
                                dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                                dst_1 = Variable(dst_1)
                                
                                for site in sites + [('Site','None')] :
                                    rlt_2 = torch.LongTensor(1,1).zero_()
                                    rlt_2[0][0] = relation_index['Site']
                                    rlt_2 = Variable(rlt_2)
                                    dst_key_2 = site[1]
                                    dst_2 = torch.LongTensor(1,1).zero_()
                                    dst_2[0][0] = e_dict[dst_type[dst_key_2]]
                                    dst_2 = Variable(dst_2)
                                    
                                    r = event_assert(src_,[(rlt_1,dst_1,rc_hidden_dict.get((src_key,dst_key_1),default_prob)),\
                                                           (rlt_2,dst_2,rc_hidden_dict.get((src_key,dst_key_2),default_prob))] )
                                    support = float(r.data[0][1] - r.data[0][0])
                                    event = Event(src_key,{('Theme',dst_key_1),('Site',dst_key_2)},support)
                                   
                                    candidate_event.append(event)
        
                        if src_type[src_key] == 'Pathway' :
                            
                            ptps = []
                            ptp2s = []
                            for para in event_para_[src_key]:
                                if para[0] == 'Participant' :
                                    ptps.append(para)
                                if para[0] == 'Participant2' :
                                    ptp2s.append(para)
                            
                            for ptp in ptps :
                                rlt_1 = torch.LongTensor(1,1).zero_()
                                rlt_1[0][0] = relation_index['Participant']
                                rlt_1 = Variable(rlt_1)
                                dst_key_1 = ptp[1]
                                dst_1 = torch.LongTensor(1,1).zero_()
                                dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                                dst_1 = Variable(dst_1)
                                
                                for ptp2 in ptp2s + [('Participant2','None')] :
                                    rlt_2 = torch.LongTensor(1,1).zero_()
                                    rlt_2[0][0] = relation_index['Participant2']
                                    rlt_2 = Variable(rlt_2)
                                    dst_key_2 = ptp2[1]
                                    dst_2 = torch.LongTensor(1,1).zero_()
                                    dst_2[0][0] = e_dict[dst_type[dst_key_2]]
                                    dst_2 = Variable(dst_2)
                                    
                                    r = event_assert(src_,[(rlt_1,dst_1,rc_hidden_dict.get((src_key,dst_key_1),default_prob)),\
                                                           (rlt_2,dst_2,rc_hidden_dict.get((src_key,dst_key_2),default_prob))] )
                                    support = float(r.data[0][1] - r.data[0][0])
                                    event = Event(src_key,{('Participant',dst_key_1),('Participant2',dst_key_2)},support)
                                    
                                    candidate_event.append(event)
               
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
                            
                            for theme in themes :
                                rlt_1 = torch.LongTensor(1,1).zero_()
                                rlt_1[0][0] = relation_index['Theme']
                                rlt_1 = Variable(rlt_1)
                                dst_key_1 = theme[1]
                                dst_1 = torch.LongTensor(1,1).zero_()
                                dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                                dst_1 = Variable(dst_1)
                                
                                for theme2 in theme2s + [('Theme2','None')] :
                                    rlt_1 = torch.LongTensor(1,1).zero_()
                                    rlt_1[0][0] = relation_index['Theme2']
                                    rlt_1 = Variable(rlt_1)
                                    dst_key_1 = theme2[1]
                                    dst_1 = torch.LongTensor(1,1).zero_()
                                    dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                                    dst_1 = Variable(dst_1)

                                    for site in sites + [('Site','None')] :
                                        rlt_3 = torch.LongTensor(1,1).zero_()
                                        rlt_3[0][0] = relation_index['Site']
                                        rlt_3 = Variable(rlt_3)
                                        dst_key_3 = site[1]
                                        dst_3 = torch.LongTensor(1,1).zero_()
                                        dst_3[0][0] = e_dict[dst_type[dst_key_3]]
                                        dst_3 = Variable(dst_3)
                                       
                                        r = event_assert(src_,[(rlt_1,dst_1,rc_hidden_dict.get((src_key,dst_key_1),default_prob)),\
                                                               (rlt_2,dst_2,rc_hidden_dict.get((src_key,dst_key_2),default_prob)),\
                                                               (rlt_3,dst_3,rc_hidden_dict.get((src_key,dst_key_3),default_prob))] )
                                        support = float(r.data[0][1] - r.data[0][0])
                                        event = Event(src_key,{('Theme',dst_key_1),('Theme2',dst_key_2),('Site',dst_key_3)},support)
                                        
                                        candidate_event.append(event)

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
                                  
                            for theme in themes :
                                rlt_1 = torch.LongTensor(1,1).zero_()
                                rlt_1[0][0] = relation_index['Theme']
                                rlt_1 = Variable(rlt_1)
                                dst_key_1 = theme[1]
                                dst_1 = torch.LongTensor(1,1).zero_()
                                dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                                dst_1 = Variable(dst_1)
                                
                                for theme2 in theme2s + [('Theme2','None')] :
                                    rlt_2 = torch.LongTensor(1,1).zero_()
                                    rlt_2[0][0] = relation_index['Theme2']
                                    rlt_2 = Variable(rlt_2)
                                    dst_key_2 = theme2[1]
                                    dst_2 = torch.LongTensor(1,1).zero_()
                                    dst_2[0][0] = e_dict[dst_type[dst_key_2]]
                                    dst_2 = Variable(dst_2)
                                    
                                    for atloc in atlocs + [('AtLoc','None')] :
                                        rlt_3 = torch.LongTensor(1,1).zero_()
                                        rlt_3[0][0] = relation_index['AtLoc']
                                        rlt_3 = Variable(rlt_3)
                                        dst_key_3 = atloc[1]
                                        dst_3 = torch.LongTensor(1,1).zero_()
                                        dst_3[0][0] = e_dict[dst_type[dst_key_3]]
                                        dst_3 = Variable(dst_3)
                                        
                                        for fromloc in fromlocs + [('FromLoc','None')] :
                                            rlt_4 = torch.LongTensor(1,1).zero_()
                                            rlt_4[0][0] = relation_index['FromLoc']
                                            rlt_4 = Variable(rlt_4)
                                            dst_key_4 = fromloc[1]
                                            dst_4 = torch.LongTensor(1,1).zero_()
                                            dst_4[0][0] = e_dict[dst_type[dst_key_4]]
                                            dst_4 = Variable(dst_4)
                                            
                                            for toloc in tolocs + [('ToLoc','None')] :
                                                rlt_5 = torch.LongTensor(1,1).zero_()
                                                rlt_5[0][0] = relation_index['ToLoc']
                                                rlt_5 = Variable(rlt_5)
                                                dst_key_5 = toloc[1]
                                                dst_5 = torch.LongTensor(1,1).zero_()
                                                dst_5[0][0] = e_dict[dst_type[dst_key_5]]
                                                dst_5 = Variable(dst_5)
                                                
                                                r = event_assert(src_,[(rlt_1,dst_1,rc_hidden_dict.get((src_key,dst_key_1),default_prob)),\
                                                                       (rlt_2,dst_2,rc_hidden_dict.get((src_key,dst_key_2),default_prob)),\
                                                                       (rlt_3,dst_3,rc_hidden_dict.get((src_key,dst_key_3),default_prob)),\
                                                                       (rlt_4,dst_4,rc_hidden_dict.get((src_key,dst_key_4),default_prob)),\
                                                                       (rlt_5,dst_5,rc_hidden_dict.get((src_key,dst_key_5),default_prob))] )
                                                support = float(r.data[0][1] - r.data[0][0])
                                                event = Event(src_key,{('Theme',dst_key_1),('Theme2',dst_key_2),('AtLoc',dst_key_3),\
                                                                       ('FromLoc',dst_key_4),('ToLoc',dst_key_5)},support)
                                                
                                                candidate_event.append(event) 
                                
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
                            
                            for theme in themes :
                                rlt_1 = torch.LongTensor(1,1).zero_()
                                rlt_1[0][0] = relation_index['Theme']
                                rlt_1 = Variable(rlt_1)
                                dst_key_1 = theme[1]
                                dst_1 = torch.LongTensor(1,1).zero_()
                                dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                                dst_1 = Variable(dst_1)
                                
                                for cause in causes + [('Cause','None')] :
                                    rlt_2 = torch.LongTensor(1,1).zero_()
                                    rlt_2[0][0] = relation_index['Cause']
                                    rlt_2 = Variable(rlt_2)
                                    dst_key_2 = cause[1]
                                    dst_2 = torch.LongTensor(1,1).zero_()
                                    dst_2[0][0] = e_dict[dst_type[dst_key_2]]
                                    dst_2 = Variable(dst_2)
                                    
                                    r = event_assert(src_,[(rlt_1,dst_1,rc_hidden_dict.get((src_key,dst_key_1),default_prob)),\
                                                           (rlt_2,dst_2,rc_hidden_dict.get((src_key,dst_key_2),default_prob))] )
                                    support = float(r.data[0][1] - r.data[0][0])
                                    event = Event(src_key,{('Theme',dst_key_1),('Cause',dst_key_2)},support)
                                    
                                    candidate_event.append(event) 
                               
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
                            
                            for theme in themes + [('Theme','None')] :
                                rlt_1 = torch.LongTensor(1,1).zero_()
                                rlt_1[0][0] = relation_index['Theme']
                                rlt_1 = Variable(rlt_1)
                                dst_key_1 = theme[1]
                                dst_1 = torch.LongTensor(1,1).zero_()
                                dst_1[0][0] = e_dict[dst_type[dst_key_1]]
                                dst_1 = Variable(dst_1)
                                
                                for theme2 in theme2s + [('Theme2','None')] :
                                    rlt_2 = torch.LongTensor(1,1).zero_()
                                    rlt_2[0][0] = relation_index['Theme2']
                                    rlt_2 = Variable(rlt_2)
                                    dst_key_2 = theme2[1]
                                    dst_2 = torch.LongTensor(1,1).zero_()
                                    dst_2[0][0] = e_dict[dst_type[dst_key_2]]
                                    dst_2 = Variable(dst_2)
                                    
                                    for instr in instrs + [('Instrument','None')] :
                                        rlt_3 = torch.LongTensor(1,1).zero_()
                                        rlt_3[0][0] = relation_index['Instrument']
                                        rlt_3 = Variable(rlt_3)
                                        dst_key_3 = instr[1]
                                        dst_3 = torch.LongTensor(1,1).zero_()
                                        dst_3[0][0] = e_dict[dst_type[dst_key_3]]
                                        dst_3 = Variable(dst_3)
                                        
                                        for instr2 in instr2s + [('Instrument2','None')] :
                                            rlt_4 = torch.LongTensor(1,1).zero_()
                                            rlt_4[0][0] = relation_index['Instrument2']
                                            rlt_4 = Variable(rlt_4)
                                            dst_key_4 = instr2[1]
                                            dst_4 = torch.LongTensor(1,1).zero_()
                                            dst_4[0][0] = e_dict[dst_type[dst_key_4]]
                                            dst_4 = Variable(dst_4)
                                            
                                            r = event_assert(src_,[(rlt_1,dst_1,rc_hidden_dict.get((src_key,dst_key_1),default_prob)),\
                                                                   (rlt_2,dst_2,rc_hidden_dict.get((src_key,dst_key_2),default_prob)),\
                                                                   (rlt_3,dst_3,rc_hidden_dict.get((src_key,dst_key_3),default_prob)),\
                                                                   (rlt_4,dst_4,rc_hidden_dict.get((src_key,dst_key_4),default_prob))] )
                                            support = float(r.data[0][1] - r.data[0][0])
                                            event = Event(src_key,{('Theme',dst_key_1),('Theme2',dst_key_2),\
                                                                   ('Instrument',dst_key_3),('Instrument2',dst_key_4)},support)
                                            
                                            candidate_event.append(event)

                        best_penalty = event_support[src_key] - 1
                        best_combination = []
                        best_range = 0
                        related_relation = dict()
                        for key in relation_support.keys() :
                            if key[0] == src_key :
                                related_relation[key] = relation_support[key]
                                best_penalty = best_penalty + related_relation[key] - 1
                        
                        for j in range(0,len(candidate_event)) :
                            best_penalty_this_round = 9999.9 # infinite max
                            penalty = 0.0
                            
                            for event in candidate_event :
                                if event.choosed == True :
                                    penalty = penalty + max(1-event.support,0.0) #-event.support*0.4
                            
                            for event in candidate_event :
                                if event.choosed == False :
                                    penalty_ = penalty + max(1-event.support,0.0) # -event.support*0.4
                                    para_nodes = event.get_all_para_nodes()
                                    for key in related_relation.keys() :
                                        if not key[1] in para_nodes :
                                            penalty_ = penalty_ + related_relation[key] - 1
                                    if penalty_ < best_penalty_this_round :
                                        best_penalty_this_round = penalty_
                                        best_event_this_round = event

                            key_list = related_relation.keys()
                            para_nodes = best_event_this_round.get_all_para_nodes()
                            for key in key_list :
                                if key[1] in para_nodes :
                                    related_relation.pop(key)
                            best_event_this_round.set_choosed()
                            best_combination.append(best_event_this_round)
                            if best_penalty_this_round < best_penalty :
                                best_penalty = best_penalty_this_round
                                best_range = j + 1

                        best_combination = best_combination[:best_range]   
                        best_combination = remove_subset(best_combination)
                        
                        for event in best_combination :
                            v_c = event.paras
                            e_id = 'E%d'%e_index
                            if not t2e.has_key(src_key) :
                                t2e[src_key] = []
                            t2e[src_key].append(e_id)
                            event_info_paras[e_id] = v_c
                            e_index = e_index + 1
                    '''
                    for t_id in set(event_loc_.keys()) - set(t2e.keys()) :
                        e_id = 'E%d'%e_index
                        t2e[t_id] = []
                        t2e[t_id].append(e_id)
                        event_info_paras[e_id] = {}
                        e_index = e_index + 1
                    '''                    
                    key_list = list(t2e.keys())
                    key_list.sort()
                    
                    for t_id in key_list : #event_info.keys():
                        info = event_info[t_id]                 
                        f.write(t_id+'\t'+info[0]+' '+str(info[1])+' '+str(info[2])+'\t'+info[3]+'\n')
                        '''
                        for e_id in t2e[t_id] :
                            t2e_t_id = t2e[t_id][:]
                            argus = event_info_paras[e_id]

                            augment_flag = False
                            for argu in argus :
                                argu_id = argu[1]
                                if argu_id in t2e.keys() :
                                    argu_ids = t2e[argu_id]
                                    if len(argu_ids) > 1 :
                                        augment_flag = True
                                        break

                            if not augment_flag :
                                continue
                            
                            argus.remove(argu)

                            for n in range(1,len(argu_ids)):
                                new_argus = argus.copy()
                                new_argus.add((argu[0],argu_ids[n]))
                                e_id_ = 'E%d'%e_index
                                e_index = e_index + 1
                                t2e_t_id.append(e_id_)
                                event_info_paras[e_id_] = new_argus

                            argus.add((argu[0],argu_ids[0]))

                        t2e[t_id] = t2e_t_id
                        '''    
                    #print key_list,t2e.keys()
                    
                    for t_id in key_list:
                        info = event_info[t_id]
                     
                        for e_id in t2e[t_id] :
                            continue_flag = False
                            argus = event_info_paras[e_id]
                            #print t_id,e_id,argus
                            
                            for argu in argus :
                                argu_id = argu[1]
                                
                                if len(argu_id) > 4 and not t2e.has_key(argu_id):
                                    continue_flag = True
                            
                            if continue_flag :
                                continue
                            
                            f.write(e_id+'\t')
                            f.write(info[0]+':'+t_id)
                         
                            for argu in argus :
                                argu_id = argu[1]
                                if argu_id == 'None' :
                                    continue
                                if argu_id in t2e.keys() :
                                    argu_id = t2e[argu_id][0] #
                                f.write(' '+argu[0]+':'+argu_id)             
                            f.write('\n')
                    
            f.close()
