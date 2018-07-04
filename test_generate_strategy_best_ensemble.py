import os
import csv
import math
import torch
import cPickle
import torch.nn.init as init
from torch.autograd import Variable
from pandas.io.parsers import read_csv
from dataloader_modified import DataLoader
from define_net import Char_CNN_encode,BiLSTM,Trigger_Recognition,Relation_ClassificationC
from sentence_set_single import Sentence_Set_Single
from define_net_event_evaluation import *
from entity_event_dict import *


class Event :
    def __init__(self, event_id, paras, support, modification):
        self.event_id = event_id
        self.paras = paras
        self.support = support
        self.modification = modification
        self.choosed = False

    def get_modification(self):
        if self.modification == 0 :
            return 'None'
        elif self.modification == 1 :
            return 'Speculation'
        elif self.modification == 2 :
            return 'Negation'

    def get_all_para_nodes(self):
        para_nodes = set()
        for para in self.paras :
            para_nodes.add(para[1])
        return para_nodes

    def set_choosed(self):
        self.choosed = True

    def reset_choosed(self):
        self.choosed = False


def remove_subset(valid_combination):
  
    valid_combination_without_none = [] # remove none-parameter
    for v_c in valid_combination :
        v_c_without_none = set()
        for para in v_c.paras :
            if not para[1] == 'None' :
                v_c_without_none.add(para)
        event = Event(v_c.event_id,v_c_without_none,v_c.support,v_c.modification)
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
    folder = path_+'/table_test/'

    ensemble = 10
    bias_tr,bias_rc = 0.5,1.0

    have_load = False
 
    for root, _, fnames in os.walk(folder):
        for fname in fnames:
            print fname
            
            testset = Sentence_Set_Single(folder+fname)
            testloader = DataLoader(testset,batch_size=1,shuffle=False)
            df = read_csv(folder+fname)

            char_dim = testset.get_char_dim()
            word_dim = testset.get_word_dim()
            entity_dim = testset.get_entity_dim()
            event_dim = testset.get_event_dim()
            relation_dim = testset.get_relation_dim()

            if not have_load :

                char_cnn_list = [ Char_CNN_encode(char_dim).cuda() for k in range(0,ensemble) ]
                bilstm_list = [ BiLSTM(word_dim,entity_dim).cuda() for k in range(0,ensemble) ]
                tr_list = [ Trigger_Recognition(event_dim).cuda() for k in range(0,ensemble) ]
                rc_list = [ Relation_ClassificationC(relation_dim).cuda() for k in range(0,ensemble) ]
                ee_list = [ Event_Evaluation().cuda() for k in range(0,ensemble) ]

                m = 50

                for k in range(0,ensemble):
                    char_cnn_list[k].load_state_dict(torch.load(path_+'/char_cnn_%d_%d.pth'%(m,k)))
                    bilstm_list[k].load_state_dict(torch.load(path_+'/bilstm_%d_%d.pth'%(m,k)))
                    tr_list[k].load_state_dict(torch.load(path_+'/tr_%d_%d.pth'%(m,k)))
                    rc_list[k].load_state_dict(torch.load(path_+'/rc_%d_%d.pth'%(m,k)))
                    ee_list[k].load_state_dict(torch.load(path_+'/event_assert_%d_%d.pth'%(m,k)))

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

                have_load = True

            t_index = 1000 # Entity number is unimportant, begin with a large number
            e_index = 1
            m_index = 0
            base_loc = 0

            result_dir = folder+fname
            result_dir = result_dir.replace('table_test','a2_result')
            result_dir = result_dir.replace('.csv','.a2')
            f = file(result_dir,'w')
            
            for i,batch in enumerate(testloader,0):
                    
                for data in batch: # due to we have modified the defination of batch, the batch here is a list

                    input_word, input_entity, input_char, target, entity_loc, event_loc, relation, _ = data
                    input_word, input_entity, target = Variable(input_word).cuda(),Variable(input_entity).cuda(),Variable(target).cuda()

                    bilstm_output_list = []
                    entity_emb_list = []
                    tr_emb_list = []
                    tr_index_list = []
                    
                    for k in range(0,ensemble):
                        char_cnn = char_cnn_list[k]
                        bilstm = bilstm_list[k]
                        tr = tr_list[k]
                        
                        char_encode = []
                        for chars in input_char :
                            chars = char_cnn(Variable(chars).cuda())
                            char_encode.append(chars)
                        char_encode = torch.cat(char_encode,0) # L*N*conv_out_channel

                        hidden1,hidden2 = bilstm.initHidden()
                        bilstm.eval()
                        bilstm_output,hidden,entity_emb = bilstm((input_word,input_entity,char_encode),(hidden1.cuda(),hidden2.cuda()))
                        bilstm_output_list.append(bilstm_output)
                        entity_emb_list.append(entity_emb)
                        
                        tr.eval()
                        tr_output,tr_emb,tr_index = tr(bilstm_output)
                        tr_emb_list.append(tr_emb)
                        tr_index_list.append(tr_index)

                        row = tr_output.data
                        
                        for x in range(0,row.size()[0]) :
                            row[x][0] = row[x][0]*1.0
                        
                        if k == 0 :
                            row_sum = row[:]
                        else :
                            for x in range(0,row.size()[0]) :
                                for y in range(0,row.size()[1]) :
                                    row_sum[x][y] = row_sum[x][y] + row[x][y]

                    tr_index = []
                    row_sum_new = []
                    for l in range(0,row_sum.size()[0]):
                        row_sum_new.append(list(row_sum[l]))
                        for j in range(0,len(row_sum_new[l])) :
                            row_sum_new[l][j] = float(row_sum_new[l][j].data)
                        row_sum_new[l][0] = row_sum_new[l][0] - bias_tr
                        index = row_sum_new[l].index(max(row_sum_new[l]))
                        tr_index.append(index)

                        for k in range(0,ensemble):
                            tr = tr_list[k]
                            vector = tr.get_event_embedding(index)
                            tr_emb_list[k][l] = vector
                        
                    previous_type = 'NONE'

                    event_loc_ = dict()
                    event_info = dict()
                    
                    for k in range(0,len(tr_index)) :
                        
                        index = tr_index[k]
                        current_type = event_index_r[index].split('-')[0]
                        if current_type == 'OTHER' :
                            current_type = 'NONE'
                        if current_type != previous_type :
                            if k != 0 and previous_type != 'NONE' : # the first one has no previous recongition
                                t_notation = 'T%d'%t_index
                                t_index = t_index + 1
                                event_info[t_notation] = ( previous_type,start_postion,end_postion,words )
                                event_loc_[t_notation] = ( start_loc,end_loc )
                            if current_type != 'NONE' : # begin a new event
                                start_loc,end_loc = k,k
                                start_postion = df['position'][k+base_loc]
                                end_postion = df['position'][k+base_loc]+len(df['word'][k+base_loc])
                                words = df['word'][k+base_loc]
                        else :
                            if current_type != 'NONE' :
                                end_loc = k
                                end_postion = df['position'][k+base_loc]+len(df['word'][k+base_loc])
                                words = words + ' ' + df['word'][k+base_loc]
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
                            index = tr_index[j]
                            support = support + float(row_sum_new[j][index])/ensemble-float(row_sum_new[j][0])/ensemble
                        support = support/(end-start+1)
                        event_support[key] = support - bias_tr

                    base_loc = base_loc+len(row_sum)
                    
                    e_loc = dict(entity_loc.items()+event_loc_.items()) # event refer to net output

                    event_para_ = dict()
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

                            for k in range(0,ensemble):

                                bilstm_output = bilstm_output_list[k]
                                entity_emb = entity_emb_list[k]
                                tr_emb = tr_emb_list[k]
                                
                                src = bilstm_output[src_begin:src_end+1]
                                src_event = tr_emb[src_begin:src_end+1]
                                dst = bilstm_output[dst_begin:dst_end+1]
                                
                                src_name = event_index_r[int(tr_index[src_begin])].split('-')[0]
                                src_type[src_key] = src_name
                                
                                event_flag = False
                                if event_loc_.has_key(dst_key):
                                    dst_event = tr_emb[dst_begin:dst_end+1]
                                    dst_name = event_index_r[int(tr_index[dst_begin])].split('-')[0]
                                    event_flag = True
                                else :
                                    dst_event = entity_emb[dst_begin:dst_end+1]
                                    dst_name = entity_index_r[int(input_entity.data[0][dst_begin])].split('-')[0]
                                dst_type[dst_key] = dst_name

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

                                rc = rc_list[k]
                                
                                rc.eval()
                                rc_output = rc(src_event,src,middle,dst,dst_event,reverse_flag,middle_flag)

                                if src_name in {'Development','Blood_vessel_development','Growth','Death',\
                                                'Cell_transformation','Metabolism','Gene_expression',\
                                                'Pathway','Localization','Planned_process'} :
                                    beta = 2.0
                                elif src_name in {'Carcinogenesis','Regulation','Phosphorylation','Metastasis',\
                                                  'Cell_death'} :
                                    beta = 3.0
                                else :
                                    beta = 5.0                                
                                row = rc_output.data
                                row[0][0] = row[0][0]*beta
                                if k == 0 :
                                    row_sum = row[:]
                                else :
                                    for x in range(0,row.size()[0]) :
                                        for y in range(0,row.size()[1]) :
                                            row_sum[x][y] = row_sum[x][y] + row[x][y]
                                
                            this_row = list(row_sum[0])
                            index = this_row.index(max(this_row))
                            current_type = relation_index_r[index]
                            current_strength = this_row[index]

                            if event_flag and not src_name in {'Regulation','Positive_regulation','Negative_regulation','Planned_process'} :
                                continue
                            if current_type == 'Site' and dst_name != 'Gene_or_gene_product':
                                continue
                            if (current_type == 'FromLoc' or current_type == 'ToLoc' or current_type == 'AtLoc') and\
                               dst_name == 'Simple_chemical':
                                continue
                            if event_flag and current_type == 'Instrument' and src_name == 'Planned_process':
                                continue
                            
                            if not event_para_.has_key(src_key) : # key point, keep the event with no relations
                                event_para_[src_key] = set()
                            
                            if current_type != 'NONE' and current_type != 'OTHER' :
                                support = (this_row[index] - this_row[0])/ensemble
                                relation_support[(src_key,dst_key)] = support - bias_rc
                                event_para_[src_key].add((current_type,dst_key))

                    dst_type['None'] = 'None'
                    e_loc['None'] = (0,0)
                    event_info_paras = dict()
                    t2e = dict()
                    e2m = dict()

                    rely_sequence = []
                    rely_count = dict()
                    for key in event_para_.keys() :
                        rely_count[key] = 0
                    for key in event_para_.keys() :
                        paras = event_para_[key]
                        for para in paras :
                            dst_key = para[1]
                            if dst_key in rely_count.keys() :
                                rely_count[dst_key] = rely_count[dst_key] + 1
                                
                    while len(rely_sequence) != len(event_para_.keys()) :
                        min_count = 9999
                        for key in rely_count.keys() :
                            if rely_count[key] < min_count :
                                min_count = rely_count[key]
                                selected_key = key
                        rely_count.pop(selected_key)
                        rely_sequence.append(selected_key)
                        for para in event_para_[selected_key] :
                            dst_key = para[1]
                            if dst_key in rely_count.keys() :
                                rely_count[dst_key] = rely_count[dst_key] - 1
                
                    for src_key in rely_sequence :

                        s_r = e_loc[src_key]
                        range_list = []
                        candidate_event = []

                        for k in range(0,ensemble):

                            bilstm_output = bilstm_output_list[k]
                            entity_emb = entity_emb_list[k]
                            tr_emb = tr_emb_list[k]

                            event_evaluation = ee_list[k]

                            support = []
                            m_list = []
                            dst_set = []
                            
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
                                    dst_key_1 = theme[1]
                                    range_list.append(('Theme',e_loc[dst_key_1]))
                                    
                                    hidden = event_evaluation.initHidden()
                                    r,m = event_evaluation(bilstm_output,tr_emb,entity_emb,s_r,range_list,hidden)
                                    
                                    support.append(float(r.data[0][1] - r.data[0][0]))
                                    m_list.append(list(m.data[0]))
                                    dst_set.append({('Theme',dst_key_1)})

                                    range_list.pop()
                                        
                            if src_type[src_key] == 'Cell_death' or \
                               src_type[src_key] == 'Amino_acid_catabolism' or \
                               src_type[src_key] == 'Glycolysis':
                                
                                themes = []
                                for para in event_para_[src_key]:
                                    if para[0] == 'Theme' :
                                        themes.append(para)
                                
                                for theme in themes + [('Theme','None')] :
                                    dst_key_1 = theme[1]
                                    range_list.append(('Theme',e_loc[dst_key_1]))
                                    
                                    hidden = event_evaluation.initHidden()
                                    r,m = event_evaluation(bilstm_output,tr_emb,entity_emb,s_r,range_list,hidden)

                                    support.append(float(r.data[0][1] - r.data[0][0]))
                                    m_list.append(list(m.data[0]))
                                    dst_set.append({('Theme',dst_key_1)})

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
                                    dst_key_1 = theme[1]
                                    range_list.append(('Theme',e_loc[dst_key_1]))
                                    
                                    for atloc in atlocs + [('AtLoc','None')] :
                                        dst_key_2 = atloc[1]
                                        range_list.append(('AtLoc',e_loc[dst_key_2]))

                                        hidden = event_evaluation.initHidden()
                                        r,m = event_evaluation(bilstm_output,tr_emb,entity_emb,s_r,range_list,hidden)
                                    
                                        support.append(float(r.data[0][1] - r.data[0][0]))
                                        m_list.append(list(m.data[0]))
                                        dst_set.append({('Theme',dst_key_1),('AtLoc',dst_key_2)})

                                        range_list.pop()
                                    range_list.pop()

                            if src_type[src_key] == 'Blood_vessel_development' or \
                               src_type[src_key] == 'Carcinogenesis' or \
                               src_type[src_key] == 'Cell_transformation':
                                
                                themes = []
                                atlocs = []
                                for para in event_para_[src_key]:
                                    if para[0] == 'Theme' :
                                        themes.append(para)
                                    if para[0] == 'AtLoc' :
                                        atlocs.append(para)
                                        
                                for theme in themes + [('Theme','None')] :
                                    dst_key_1 = theme[1]
                                    range_list.append(('Theme',e_loc[dst_key_1]))
                                    
                                    for atloc in atlocs + [('AtLoc','None')] :
                                        dst_key_2 = atloc[1]
                                        range_list.append(('AtLoc',e_loc[dst_key_2]))

                                        hidden = event_evaluation.initHidden()
                                        r,m = event_evaluation(bilstm_output,tr_emb,entity_emb,s_r,range_list,hidden)

                                        support.append(float(r.data[0][1] - r.data[0][0]))
                                        m_list.append(list(m.data[0]))
                                        dst_set.append({('Theme',dst_key_1),('AtLoc',dst_key_2)})
                                        
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
                                    dst_key_1 = theme[1]
                                    range_list.append(('Theme',e_loc[dst_key_1]))
                                    
                                    for atloc in atlocs + [('AtLoc','None')] :
                                        dst_key_2 = atloc[1]
                                        range_list.append(('AtLoc',e_loc[dst_key_2]))
                                    
                                        for site in sites + [('Site','None')] :
                                            dst_key_3 = site[1]
                                            range_list.append(('Site',e_loc[dst_key_3]))

                                            hidden = event_evaluation.initHidden()
                                            r,m = event_evaluation(bilstm_output,tr_emb,entity_emb,s_r,range_list,hidden)
                                    
                                            support.append(float(r.data[0][1] - r.data[0][0]))
                                            m_list.append(list(m.data[0]))
                                            dst_set.append({('Theme',dst_key_1),('AtLoc',dst_key_2),('Site',dst_key_3)})
                                            
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
                                    dst_key_1 = theme[1]
                                    range_list.append(('Theme',e_loc[dst_key_1]))
                                    
                                    for toloc in tolocs + [('ToLoc','None')]:
                                        dst_key_2 = toloc[1]
                                        range_list.append(('ToLoc',e_loc[dst_key_2]))

                                        hidden = event_evaluation.initHidden()
                                        r,m = event_evaluation(bilstm_output,tr_emb,entity_emb,s_r,range_list,hidden)
                                        
                                        support.append(float(r.data[0][1] - r.data[0][0]))
                                        m_list.append(list(m.data[0]))
                                        dst_set.append({('Theme',dst_key_1),('ToLoc',dst_key_2)})
                                        
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
                                    dst_key_1 = theme[1]
                                    range_list.append(('Theme',e_loc[dst_key_1]))
                                    
                                    for ptp in ptps + [('Participant','None')] :
                                        dst_key_2 = ptp[1]
                                        range_list.append(('Participant',e_loc[dst_key_2]))

                                        hidden = event_evaluation.initHidden()
                                        r,m = event_evaluation(bilstm_output,tr_emb,entity_emb,s_r,range_list,hidden)
                                    
                                        support.append(float(r.data[0][1] - r.data[0][0]))
                                        m_list.append(list(m.data[0]))
                                        dst_set.append({('Theme',dst_key_1),('Participant',dst_key_2)})
                                        
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
                                    dst_key_1 = theme[1]
                                    range_list.append(('Theme',e_loc[dst_key_1]))
                                    
                                    for theme2 in theme2s + [('Theme2','None')] :
                                        dst_key_2 = theme2[1]
                                        range_list.append(('Theme2',e_loc[dst_key_2]))

                                        hidden = event_evaluation.initHidden()
                                        r,m = event_evaluation(bilstm_output,tr_emb,entity_emb,s_r,range_list,hidden)
                                    
                                        support.append(float(r.data[0][1] - r.data[0][0]))
                                        m_list.append(list(m.data[0]))
                                        dst_set.append({('Theme',dst_key_1),('Theme2',dst_key_2)})

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
                                        
                                for theme in themes :
                                    dst_key_1 = theme[1]
                                    range_list.append(('Theme',e_loc[dst_key_1]))
                                    
                                    for site in sites + [('Site','None')] :
                                        dst_key_2 = site[1]
                                        range_list.append(('Site',e_loc[dst_key_2]))

                                        hidden = event_evaluation.initHidden()
                                        r,m = event_evaluation(bilstm_output,tr_emb,entity_emb,s_r,range_list,hidden)
                                    
                                        support.append(float(r.data[0][1] - r.data[0][0]))
                                        m_list.append(list(m.data[0]))
                                        dst_set.append({('Theme',dst_key_1),('Site',dst_key_2)})

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

                                for theme in themes + [('Theme','None')] :
                                    dst_key_1 = theme[1]
                                    range_list.append(('Theme',e_loc[dst_key_1]))
                                    
                                    for ptp in ptps + [('Participant','None')] :
                                        dst_key_2 = ptp[1]
                                        range_list.append(('Participant',e_loc[dst_key_2]))
                                        
                                        for ptp2 in ptp2s + [('Participant2','None')] :
                                            dst_key_3 = ptp2[1]
                                            range_list.append(('Participant2',e_loc[dst_key_3]))

                                            hidden = event_evaluation.initHidden()
                                            r,m = event_evaluation(bilstm_output,tr_emb,entity_emb,s_r,range_list,hidden)
                                        
                                            support.append(float(r.data[0][1] - r.data[0][0]))
                                            m_list.append(list(m.data[0]))
                                            dst_set.append({('Theme',dst_key_1),('Participant',dst_key_2),('Participant2',dst_key_3)})
                                            
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
                                        
                                for theme in themes :
                                    dst_key_1 = theme[1]
                                    range_list.append(('Theme',e_loc[dst_key_1]))
                                    
                                    for theme2 in theme2s + themes + [('Theme2','None')] :
                                        if theme == theme2 :
                                            continue
                                        dst_key_2 = theme2[1]
                                        range_list.append(('Theme2',e_loc[dst_key_2]))
                                    
                                        for site in sites + [('Site','None')] :
                                            dst_key_3 = site[1]
                                            range_list.append(('Site',e_loc[dst_key_3]))

                                            hidden = event_evaluation.initHidden()
                                            r,m = event_evaluation(bilstm_output,tr_emb,entity_emb,s_r,range_list,hidden)
                                    
                                            support.append(float(r.data[0][1] - r.data[0][0]))
                                            m_list.append(list(m.data[0]))
                                            dst_set.append({('Theme',dst_key_1),('Theme2',dst_key_2),('Site',dst_key_3)})
                                            
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
                                        
                                for theme in themes :
                                    dst_key_1 = theme[1]
                                    range_list.append(('Theme',e_loc[dst_key_1]))
                                    
                                    for theme2 in theme2s + [('Theme2','None')] :
                                        dst_key_2 = theme2[1]
                                        range_list.append(('Theme2',e_loc[dst_key_2]))
                                    
                                        for atloc in atlocs + [('AtLoc','None')] :
                                            dst_key_3 = atloc[1]
                                            range_list.append(('AtLoc',e_loc[dst_key_3]))
                                    
                                            for fromloc in fromlocs + [('FromLoc','None')] :
                                                dst_key_4 = fromloc[1]
                                                range_list.append(('FromLoc',e_loc[dst_key_4]))
                                    
                                                for toloc in tolocs + [('ToLoc','None')] :
                                                    dst_key_5 = toloc[1]
                                                    range_list.append(('ToLoc',e_loc[dst_key_5]))
                                                    
                                                    hidden = event_evaluation.initHidden()
                                                    r,m = event_evaluation(bilstm_output,tr_emb,entity_emb,s_r,range_list,hidden)
                                            
                                                    support.append(float(r.data[0][1] - r.data[0][0]))
                                                    m_list.append(list(m.data[0]))
                                                    dst_set.append({('Theme',dst_key_1),('Theme2',dst_key_2),('AtLoc',dst_key_3),\
                                                                    ('FromLoc',dst_key_4),('ToLoc',dst_key_5)})
                                                    
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
                                        
                                for theme in themes :
                                    dst_key_1 = theme[1]
                                    range_list.append(('Theme',e_loc[dst_key_1]))
                                    
                                    for cause in causes + [('Cause','None')] :
                                        dst_key_2 = cause[1]
                                        range_list.append(('Cause',e_loc[dst_key_2]))

                                        hidden = event_evaluation.initHidden()
                                        r,m = event_evaluation(bilstm_output,tr_emb,entity_emb,s_r,range_list,hidden)
                                    
                                        support.append(float(r.data[0][1] - r.data[0][0]))
                                        m_list.append(list(m.data[0]))
                                        dst_set.append({('Theme',dst_key_1),('Cause',dst_key_2)})

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

                                for theme in themes + [('Theme','None')] :
                                    dst_key_1 = theme[1]
                                    range_list.append(('Theme',e_loc[dst_key_1]))
                                    
                                    for theme2 in theme2s + [('Theme2','None')] :
                                        dst_key_2 = theme2[1]
                                        range_list.append(('Theme2',e_loc[dst_key_2]))
                                    
                                        for instr in instrs + [('Instrument','None')] :
                                            dst_key_3 = instr[1]
                                            range_list.append(('Instrument',e_loc[dst_key_3]))
                                    
                                            for instr2 in instr2s + [('Instrument2','None')] :
                                                dst_key_4 = instr2[1]
                                                range_list.append(('Instrument2',e_loc[dst_key_4]))

                                                hidden = event_evaluation.initHidden()
                                                r,m = event_evaluation(bilstm_output,tr_emb,entity_emb,s_r,range_list,hidden)
                                                
                                                support.append(float(r.data[0][1] - r.data[0][0]))
                                                m_list.append(list(m.data[0]))
                                                dst_set.append({('Theme',dst_key_1),('Theme2',dst_key_2),\
                                                                ('Instrument',dst_key_3),('Instrument2',dst_key_4)})

                                                range_list.pop()
                                            range_list.pop()
                                        range_list.pop()
                                    range_list.pop()

                            if k == 0 :
                                support_sum = support[:]
                                m_list_sum = m_list[:]
                            else :
                                support_sum = [ support_sum[j] + support[j] for j in range(0,len(support_sum)) ]
                                m_list_sum = [ [ m_list_sum[j][jj]+m_list[j][jj] for jj in range(0,3) ] for j in range(0,len(m_list_sum)) ]

                        for j in range(0,len(support_sum)) :
                            m_list_sum[j][0] = m_list_sum[j][0] * 4.0
                            m_index_ = m_list_sum[j].index(max(m_list_sum[j]))
                            event = Event(src_key,dst_set[j],support_sum[j]/ensemble,m_index_)
                            candidate_event.append(event)

                        best_penalty = event_support[src_key]*1.5
                        best_combination = []
                        best_range = 0
                        related_relation = dict()
                        for key in relation_support.keys() :
                            if key[0] == src_key :
                                related_relation[key] = relation_support[key]
                                best_penalty = best_penalty + related_relation[key]*0.2
                        additional_penalty = best_penalty
                        
                        for j in range(0,len(candidate_event)) :
                            best_penalty_this_round = 9999.9 # infinite max
                            penalty = 0.0
                                    
                            for event in candidate_event :
                                if event.choosed == True :
                                    penalty = penalty + max(2.0-event.support,0.0) #-event.support*0.4
                                    
                            for event in candidate_event :
                                if event.choosed == False :
                                    penalty_ = penalty + max(2.0-event.support,0.0) # -event.support*0.4
                                    para_nodes = event.get_all_para_nodes()
                                    for key in related_relation.keys() :
                                        if not key[1] in para_nodes :
                                            penalty_ = penalty_ + related_relation[key]*0.2
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
                        #best_combination = remove_subset(best_combination)
                        additional_penalty = additional_penalty - best_penalty
                                
                        for event in best_combination :
                            v_c = event.paras
                            e_id = 'E%d'%e_index
                            if not t2e.has_key(src_key) :
                                t2e[src_key] = []
                            t2e[src_key].append(e_id)
                            e2m[e_id] = event.get_modification()
                            event_info_paras[e_id] = v_c
                            e_index = e_index + 1

                            for para in v_c :
                                dst_key = para[1]
                                if event_support.has_key(dst_key) :
                                    event_support[src_key] = event_support[src_key] + additional_penalty
                            
                    t_key_list = list(t2e.keys())
                    t_key_list.sort()
                    
                    for t_id in t_key_list : #event_info.keys():
                        info = event_info[t_id]
                        string = info[3]
                        string = string.replace(' - ','-')
                        string = string.replace(' -','-')
                        string = string.replace('- ','-')
                        string = string.replace(' . ','.')
                        string = string.replace(' .','.')
                        string = string.replace('. ','.')
                        string = string.replace(' ( ','(')
                        string = string.replace(' (','(')
                        string = string.replace('( ','(')
                        string = string.replace(' ) ',')')
                        string = string.replace(' )',')')
                        string = string.replace(') ',')')
                        string = string.replace(' / ','/')
                        string = string.replace(' /','/')
                        string = string.replace('/ ','/')
                        f.write(t_id+'\t'+info[0]+' '+str(info[1])+' '+str(info[2])+'\t'+string+'\n')
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
                                break
                                
                            argus.remove(argu)

                            for n in range(1,len(argu_ids)):
                                new_argus = argus.copy()
                                new_argus.add((argu[0],argu_ids[n]))
                                e_id_ = 'E%d'%e_index
                                e_index = e_index + 1
                                t2e_t_id.append(e_id_)
                                e2m[e_id_] = e2m[e_id]
                                event_info_paras[e_id_] = new_argus

                            argus.add((argu[0],argu_ids[0]))

                        t2e[t_id] = t2e_t_id
                        '''
                    unstable = True
                    while unstable :
                        unstable = False
                        e_key_list = event_info_paras.keys()

                        for e_id in e_key_list:
                            t_key_list = t2e.keys()
                            argus = event_info_paras[e_id]
                            delete = False
                            for argu in argus:
                                argu_id = argu[1]
                                if (len(argu_id) == 5) and (argu_id not in t_key_list) :
                                    delete = True
                            if delete :
                                for key in t_key_list :
                                    for e_id_ in t2e[key] :
                                        if e_id_ == e_id :
                                            t2e[key].remove(e_id_)
                                    if len(t2e[key]) == 0:
                                        t2e.pop(key)
                                event_info_paras.pop(e_id)
                                unstable = True
                                
                    e2e = dict()
                    for t_id in t2e.keys():
                        info = event_info[t_id]
                     
                        for e_id in t2e[t_id] :
                            continue_flag = False
                            argus = event_info_paras[e_id]
                            e2e[e_id] = set()
                            
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
                                    e2e[e_id].add(argu_id)
                                    if e_id in e2e.get(argu_id,set()) :
                                        print 'recursion!' 
                                f.write(' '+argu[0]+':'+argu_id)             
                            f.write('\n')

                            if e2m[e_id] != 'None' :
                                m_index = m_index + 1
                                f.write('M%d'%m_index+'\t'+e2m[e_id]+' '+e_id+'\n')
                    
            f.close()
