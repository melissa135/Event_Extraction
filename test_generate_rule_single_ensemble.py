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


def illegal_argument_cut(etype,argus):

    if etype == 'Development' :
        for argu in argus.keys() :
            if not argu in { 'Theme' } :
                argus.pop(argu)
    elif etype == 'Blood_vessel_development' :
        for argu in argus.keys() :
            if not argu in { 'Theme','AtLoc' }:
                argus.pop(argu)
    elif etype == 'Growth' :
        for argu in argus.keys() :
            if not argu in { 'Theme' }:
                argus.pop(argu)
    elif etype == 'Death' :
        for argu in argus.keys() :
            if not argu in { 'Theme' }:
                argus.pop(argu)
    elif etype == 'Cell_death' :
        for argu in argus.keys() :
            if not argu in { 'Theme' }:
                argus.pop(argu)
    elif etype == 'Breakdown' :
        for argu in argus.keys() :
            if not argu in { 'Theme' }:
                argus.pop(argu)
    elif etype == 'Cell_proliferation' :
        for argu in argus.keys() :
            if not argu in { 'Theme' }:
                argus.pop(argu)
    elif etype == 'Cell_division' :
        for argu in argus.keys() :
            if not argu in { 'Theme' }:
                argus.pop(argu)
    elif etype == 'Cell_differentiation' :
        for argu in argus.keys() :
            if not argu in { 'Theme','AtLoc' }:
                argus.pop(argu)
    elif etype == 'Remodeling' :
        for argu in argus.keys() :
            if not argu in { 'Theme' }:
                argus.pop(argu)
    elif etype == 'Reproduction' :
        for argu in argus.keys() :
            if not argu in { 'Theme' }:
                argus.pop(argu)
    elif etype == 'Mutation' :
        for argu in argus.keys() :
            if not argu in { 'Theme','AtLoc','Site' }:
                argus.pop(argu)
    elif etype == 'Carcinogenesis' :
        for argu in argus.keys() :
            if not argu in { 'Theme','AtLoc' }:
                argus.pop(argu)
    elif etype == 'Cell_transformation' :
        for argu in argus.keys() :
            if not argu in { 'Theme','AtLoc' }:
                argus.pop(argu)
    elif etype == 'Metastasis' :
        for argu in argus.keys() :
            if not argu in { 'Theme','ToLoc' }:
                argus.pop(argu)
    elif etype == 'Infection' :
        for argu in argus.keys() :
            if not argu in { 'Theme','Participant' }:
                argus.pop(argu)
    elif etype == 'Metabolism' :
        for argu in argus.keys() :
            if not argu in { 'Theme' }:
                argus.pop(argu)
    elif etype == 'Synthesis' :
        for argu in argus.keys() :
            if not argu in { 'Theme' }:
                argus.pop(argu)
    elif etype == 'Catabolism' :
        for argu in argus.keys() :
            if not argu in { 'Theme' }:
                argus.pop(argu)
    elif etype == 'Amino_acid_catabolism' :
        for argu in argus.keys() :
            if not argu in { 'Theme' }:
                argus.pop(argu)
    elif etype == 'Glycolysis' :
        for argu in argus.keys() :
            if not argu in { 'Theme' }:
                argus.pop(argu)
    elif etype == 'Gene_expression' :
        for argu in argus.keys() :
            if not argu in { 'Theme' }:
                argus.pop(argu)
    elif etype == 'Transcription' :
        for argu in argus.keys() :
            if not argu in { 'Theme' }:
                argus.pop(argu)
    elif etype == 'Translation' :
        for argu in argus.keys() :
            if not argu in { 'Theme' }:
                argus.pop(argu)
    elif etype == 'Protein_processing' :
        for argu in argus.keys() :
            if not argu in { 'Theme' }:
                argus.pop(argu)
    elif etype == 'Phosphorylation' :
        for argu in argus.keys() :
            if not argu in { 'Theme','Site' }:
                argus.pop(argu)
    elif etype == 'Ubiquitination' :
        for argu in argus.keys() :
            if not argu in { 'Theme','Site' }:
                argus.pop(argu)
    elif etype == 'Dephosphorylation' :
        for argu in argus.keys() :
            if not argu in { 'Theme','Site' }:
                argus.pop(argu)
    elif etype == 'DNA_demethylation' :
        for argu in argus.keys() :
            if not argu in { 'Theme','Site' }:
                argus.pop(argu)
    elif etype == 'Glycosylation' :
        for argu in argus.keys() :
            if not argu in { 'Theme','Site' }:
                argus.pop(argu)
    elif etype == 'Pathway' :
        for argu in argus.keys() :
            if not argu in { 'Participant' }:
                argus.pop(argu)
    elif etype == 'Binding' :
        for argu in argus.keys() :
            if not argu in { 'Theme','Site' }:
                argus.pop(argu)
    elif etype == 'Dissociation' :
        for argu in argus.keys() :
            if not argu in { 'Theme','Site' }:
                argus.pop(argu)
    elif etype == 'Localization' :
        for argu in argus.keys() :
            if not argu in { 'Theme','AtLoc','FromLoc','ToLoc' }:
                argus.pop(argu)
    elif etype == 'Regulation' :
        for argu in argus.keys() :
            if not argu in { 'Theme','Cause' }:
                argus.pop(argu)
    elif etype == 'Positive_regulation' :
        for argu in argus.keys() :
            if not argu in { 'Theme','Cause' }:
                argus.pop(argu)
    elif etype == 'Negative_regulation' :
        for argu in argus.keys() :
            if not argu in { 'Theme','Cause' }:
                argus.pop(argu)
    elif etype == 'Planned_process' :
        for argu in argus.keys() :
            if not argu in { 'Theme','Instrument' }:
                argus.pop(argu)
    return argus


def missing_argument_fill(t_id,etype,argus,e_loc,entity_loc):
    
    left = e_loc[t_id][0]
    right = e_loc[t_id][1]
    used_id = set()
    used_id.add(t_id)
    for key in argus.keys() :
        argu_id = argus[key][0]
        used_id.add(argu_id)
    # get nearest entity and event
    min_distance = 9999
    nearest_id = 'None'
    for e_id in e_loc.keys() :
        if e_id in used_id :
            continue

        begin = e_loc[e_id][0]
        end = e_loc[e_id][1]
        if begin > right :
            distance = begin - right
        else :
            distance = left - end
        if distance < min_distance :
            nearest_id = e_id
            min_distance = distance
    
    if nearest_id == 'None' :
        return argus
    argu_tuple = (nearest_id,0) ###

    if etype == 'Regulation' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = argu_tuple
    elif etype == 'Positive_regulation' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = argu_tuple
    elif etype == 'Negative_regulation' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = argu_tuple

    # get nearest entity
    min_distance = 9999
    nearest_id_et = 'None'
    for e_id in entity_loc.keys() :
        if e_id in used_id :
            continue

        begin = entity_loc[e_id][0]
        end = entity_loc[e_id][1]
        if begin > right :
            distance = begin - right
        else :
            distance = left - end
        if distance < min_distance :
            nearest_id_et = e_id
            min_distance = distance

    if nearest_id_et == 'None' :
        return argus
    argu_tuple = (nearest_id_et,0) ###

    if etype == 'Development' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = argu_tuple
    elif etype == 'Growth' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = argu_tuple
    elif etype == 'Death' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = argu_tuple
    elif etype == 'Breakdown' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = argu_tuple
    elif etype == 'Cell_proliferation' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = argu_tuple
    elif etype == 'Cell_division' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = argu_tuple
    elif etype == 'Cell_differentiation' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = argu_tuple
    elif etype == 'Remodeling' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = argu_tuple
    elif etype == 'Reproduction' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = argu_tuple
    elif etype == 'Mutation' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = argu_tuple
    elif etype == 'Cell_transformation' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = argu_tuple
    #elif etype == 'Metastasis' :
    #    if not 'ToLoc' in argus.keys() :
    #        argus['ToLoc'] = argu_tuple
    elif etype == 'Metabolism' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = argu_tuple
    elif etype == 'Synthesis' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = argu_tuple
    elif etype == 'Catabolism' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = argu_tuple
    elif etype == 'Gene_expression' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = argu_tuple
    elif etype == 'Transcription' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = argu_tuple
    elif etype == 'Translation' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = argu_tuple
    elif etype == 'Protein_processing' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = argu_tuple
    elif etype == 'Phosphorylation' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = argu_tuple
    elif etype == 'Ubiquitination' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = argu_tuple
    elif etype == 'Dephosphorylation' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = argu_tuple
    elif etype == 'DNA_demethylation' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = argu_tuple
    elif etype == 'Acetylation' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = argu_tuple
    elif etype == 'DNA_methylation' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = argu_tuple
    elif etype == 'Glycosylation' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = argu_tuple
    #elif etype == 'Pathway' :
    #    if not 'Participant' in argus.keys() :
    #        argus['Participant'] = argu_tuple
    elif etype == 'Binding' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = argu_tuple
    elif etype == 'Dissociation' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = argu_tuple
    elif etype == 'Localization' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = argu_tuple
    
    return argus


def incomplete_event_assert(etype,argus):
    
    if etype == 'Development' :
        if not 'Theme' in argus.keys() :
            return False
    elif etype == 'Growth' :
        if not 'Theme' in argus.keys() :
            return False
    elif etype == 'Death' :
        if not 'Theme' in argus.keys() :
            return False
    elif etype == 'Breakdown' :
        if not 'Theme' in argus.keys() :
            return False
    elif etype == 'Cell_proliferation' :
        if not 'Theme' in argus.keys() :
            return False
    elif etype == 'Cell_division' :
        if not 'Theme' in argus.keys() :
            return False
    elif etype == 'Cell_differentiation' :
        if not 'Theme' in argus.keys() :
            return False
    elif etype == 'Remodeling' :
        if not 'Theme' in argus.keys() :
            return False
    elif etype == 'Reproduction' :
        if not 'Theme' in argus.keys() :
            return False
    elif etype == 'Mutation' :
        if not 'Theme' in argus.keys() :
            return False
    elif etype == 'Cell_transformation' :
        if not 'Theme' in argus.keys() :
            return False
    elif etype == 'Metastasis' :
        if not 'ToLoc' in argus.keys() :
            return False
    elif etype == 'Metabolism' :
        if not 'Theme' in argus.keys() :
            return False
    elif etype == 'Synthesis' :
        if not 'Theme' in argus.keys() :
            return False
    elif etype == 'Catabolism' :
        if not 'Theme' in argus.keys() :
            return False
    elif etype == 'Gene_expression' :
        if not 'Theme' in argus.keys() :
            return False
    elif etype == 'Transcription' :
        if not 'Theme' in argus.keys() :
            return False
    elif etype == 'Translation' :
        if not 'Theme' in argus.keys() :
            return False
    elif etype == 'Protein_processing' :
        if not 'Theme' in argus.keys() :
            return False
    elif etype == 'Phosphorylation' :
        if not 'Theme' in argus.keys() :
            return False
    elif etype == 'Ubiquitination' :
        if not 'Theme' in argus.keys() :
            return False
    elif etype == 'Dephosphorylation' :
        if not 'Theme' in argus.keys() :
            return False
    elif etype == 'DNA_demethylation' :
        if not 'Theme' in argus.keys() :
            return False
    elif etype == 'Acetylation' :
        if not 'Theme' in argus.keys() :
            return False
    elif etype == 'DNA_methylation' :
        if not 'Theme' in argus.keys() :
            return False
    elif etype == 'Glycosylation' :
        if not 'Theme' in argus.keys() :
            return False
    elif etype == 'Pathway' :
        if not 'Participant' in argus.keys() :
            return False
    elif etype == 'Binding' :
        if not 'Theme' in argus.keys() :
            return False
    elif etype == 'Dissociation' :
        if not 'Theme' in argus.keys() :
            return False
    elif etype == 'Localization' :
        if not 'Theme' in argus.keys() :
            return False
    elif etype == 'Regulation' :
        if not 'Theme' in argus.keys() :
            return False
    elif etype == 'Positive_regulation' :
        if not 'Theme' in argus.keys() :
            return False
    elif etype == 'Negative_regulation' :
        if not 'Theme' in argus.keys() :
            return False
    return True


if __name__ == '__main__':

    path_ = os.path.abspath('.')
    folder = path_+'/table_test/'

    ensemble = 10

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
                
                m = 50

                for k in range(0,ensemble):
                    char_cnn_list[k].load_state_dict(torch.load(path_+'/char_cnn_%d_%d.pth'%(m,k)))
                    bilstm_list[k].load_state_dict(torch.load(path_+'/bilstm_%d_%d.pth'%(m,k)))
                    tr_list[k].load_state_dict(torch.load(path_+'/ner_%d_%d.pth'%(m,k)))
                    rc_list[k].load_state_dict(torch.load(path_+'/rc_%d_%d.pth'%(m,k)))
                    
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
                            row[x][0] = row[x][0]*1.5
                        
                        if k == 0 :
                            row_sum = row[:]
                        else :
                            for x in range(0,row.size()[0]) :
                                for y in range(0,row.size()[1]) :
                                    row_sum[x][y] = row_sum[x][y] + row[x][y]

                    previous_type = 'NONE'

                    event_loc_ = dict()
                    event_info = dict()
                    event_argu = dict()
                    tr_index_mean = []
                    
                    for k in range(0,len(row_sum)) :
                        
                        this_row = list(row_sum[k])
                        index = this_row.index(max(this_row))
                        tr_index_mean.append(index)
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

                    base_loc = base_loc+len(row_sum)
                    
                    e_loc = dict(entity_loc.items()+event_loc_.items()) # event refer to net output
                        
                    for src_key in event_loc_.keys() :
                        relations = dict()
                       
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
                                tr_index = tr_index_list[k]
                                
                                src = bilstm_output[src_begin:src_end+1]
                                for j in range(src_begin,src_end+1) :
                                    vector = tr_list[k].get_event_embedding(tr_index_mean[j])
                                    tr_emb[j] = vector
                                src_event = tr_emb[src_begin:src_end+1]
                                dst = bilstm_output[dst_begin:dst_end+1]
                                
                                src_name = event_index_r[int(tr_index_mean[src_begin])].split('-')[0]
                                event_flag = False
                                if event_loc_.has_key(dst_key):
                                    for j in range(dst_begin,dst_end+1) :
                                        vector = tr_list[k].get_event_embedding(tr_index_mean[j])
                                        tr_emb[j] = vector
                                    dst_event = tr_emb[dst_begin:dst_end+1]
                                    dst_name = event_index_r[int(tr_index_mean[dst_begin])].split('-')[0]
                                    event_flag = True
                                else :
                                    dst_event = entity_emb[dst_begin:dst_end+1]
                                    dst_name = entity_index_r[int(input_entity.data[0][dst_begin])].split('-')[0]

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

                                row = rc_output.data
                               
                                row[0][0] = row[0][0]*3.0
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
                            if event_flag and current_type == 'Instrument' and src_name == 'Planned_process':
                                continue
                            
                            if current_type != 'NONE' and current_type != 'OTHER' :
                                
                                if not relations.has_key(current_type) :
                                    relations[current_type] = (dst_key,current_strength)
                                else :
                                    strength = relations[current_type][1]
                                    if current_strength > strength :
                                        relations[current_type] = (dst_key,current_strength)
                            
                        event_argu[src_key] = dict(relations)

                key_list = list(event_loc_.keys())
                key_list.sort()
                e_loc = dict(entity_loc.items()+event_loc_.items()) #
                t2e = dict()
                
                for t_id in key_list:
                    
                    info = event_info[t_id]
                    
                    argus = illegal_argument_cut(info[0],event_argu[t_id])
                    argus = missing_argument_fill(t_id,info[0],event_argu[t_id],e_loc,entity_loc)
                    if not incomplete_event_assert(info[0],argus) :
                        event_argu.pop(t_id)
                        continue

                    event_argu[t_id] = argus
                    
                    e_id = 'E%d'%e_index
                    e_index = e_index + 1
                    t2e[t_id] = e_id                   
                    string = info[3]
                    string = string.replace(' - ','-')
                    string = string.replace(' -','-')
                    string = string.replace('- ','-')
                    string = string.replace(' . ','.')
                    string = string.replace(' .','.')
                    string = string.replace('. ','.')
                    f.write(t_id+'\t'+info[0]+' '+str(info[1])+' '+str(info[2])+'\t'+string+'\n')

                unstable = True
                while unstable :
                    unstable = False
                    key_list = event_argu.keys()
                    for t_id in key_list:
                        argus = event_argu[t_id]
                        delete = False
                        for argu_type in argus.keys():
                            argu_id = argus[argu_type][0]
                            if (len(argu_id) == 5) and (argu_id not in event_argu.keys()) :
                                delete = True
                        if delete :
                            event_argu.pop(t_id)
                            unstable = True

                for t_id in event_argu.keys():
                    
                    e_id = t2e[t_id]
                    argus = event_argu[t_id]

                    flag = True
                    for argu_type in argus.keys():
                        argu_id = argus[argu_type][0]
                        if (argu_id[0] == 'T') and (len(argu_id) == 5) and (argu_id not in t2e.keys()) :
                            flag = False
                    if not flag :
                        continue

                    f.write(e_id+'\t')
                    f.write(event_info[t_id][0]+':'+t_id)
                    
                    for argu_type in argus.keys():
                        argu_id = argus[argu_type][0]
                        if argu_id in t2e.keys() :
                            argu_id = t2e[argu_id]
                        if argu_id[0] == 'T' and len(argu_id) == 5 :
                            print argu_id
                        f.write(' '+argu_type+':'+argu_id)             
                    f.write('\n')

            f.close()
