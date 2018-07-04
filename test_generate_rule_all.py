import os
import csv
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
    elif etype == 'Protein_procession' :
        for argu in argus.keys() :
            if not argu in { 'Theme' }:
                argus.pop(argu)
    elif etype == 'Phosphorylation' :
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


def missing_argument_fill(t_id,etype,argus,e_loc):
    # get nearest entity
    left = e_loc[t_id][0]
    right = e_loc[t_id][1]
    used_id = set()
    used_id.add(t_id)
    for key in argus.keys() :
        argu_id = argus[key][0]
        used_id.add(argu_id)
    
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
    
    if etype == 'Development' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = {argu_tuple}
    elif etype == 'Growth' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = {argu_tuple}
    elif etype == 'Death' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = {argu_tuple}
    elif etype == 'Breakdown' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = {argu_tuple}
    elif etype == 'Cell_proliferation' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = {argu_tuple}
    elif etype == 'Cell_division' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = {argu_tuple}
    elif etype == 'Cell_differentiation' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = {argu_tuple}
    elif etype == 'Remodeling' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = {argu_tuple}
    elif etype == 'Reproduction' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = {argu_tuple}
    elif etype == 'Mutation' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = {argu_tuple}
    elif etype == 'Cell_transformation' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = {argu_tuple}
    #elif etype == 'Metastasis' :
    #    if not 'ToLoc' in argus.keys() :
    #        argus['ToLoc'] = argu_tuple
    elif etype == 'Metabolism' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = {argu_tuple}
    elif etype == 'Synthesis' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = {argu_tuple}
    elif etype == 'Catabolism' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = {argu_tuple}
    elif etype == 'Gene_expression' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = {argu_tuple}
    elif etype == 'Transcription' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = {argu_tuple}
    elif etype == 'Translation' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = {argu_tuple}
    elif etype == 'Protein_procession' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = {argu_tuple}
    elif etype == 'Phosphorylation' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = {argu_tuple}
    #elif etype == 'Pathway' :
    #    if not 'Participant' in argus.keys() :
    #        argus['Participant'] = argu_tuple
    elif etype == 'Binding' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = {argu_tuple}
    elif etype == 'Localization' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = {argu_tuple}
    elif etype == 'Regulation' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = {argu_tuple}
    elif etype == 'Positive_regulation' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = {argu_tuple}
    elif etype == 'Negative_regulation' :
        if not 'Theme' in argus.keys() :
            argus['Theme'] = {argu_tuple}
    return argus


def getrate_all_combination(relations) :
    
    key_list = relations.keys()

    if len(key_list) == 0 :
        return []
    
    if len(key_list) == 1 :
        key = key_list[0]
        items = relations[key]
        result = []
        for item in items :
            result.append([(key,item[0])])
        
    else :
        key = key_list.pop()
        items = relations[key]
        del relations[key]
        sub_result = getrate_all_combination(relations)
        result = []
        for item in items :
            for sub_set in sub_result :
                new_set = sub_set[:]
                new_set.append((key,item[0]))
                result.append(new_set)

    return result


if __name__ == '__main__':

    path_ = os.path.abspath('.')
    folder = '/home/zhu/event_extraction/table_test/'

    first_load = True

    global_count = 0
    
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

                char_cnn = Char_CNN_encode(char_dim).cuda()
                bilstm = BiLSTM(word_dim,entity_dim).cuda()
                tr = Trigger_Recognition(event_dim).cuda()
                rc = Relation_ClassificationC(relation_dim).cuda()

                epoch = 10
                number = 0
                char_cnn.load_state_dict(torch.load(path_+'/char_cnn_%d_%d.pth'%(epoch,number)))
                bilstm.load_state_dict(torch.load(path_+'/bilstm_%d_%d.pth'%(epoch,number)))
                tr.load_state_dict(torch.load(path_+'/ner_%d_%d.pth'%(epoch,number)))
                rc.load_state_dict(torch.load(path_+'/rc_%d_%d.pth'%(epoch,number)))

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

                    input_word, input_entity, input_char, target, entity_loc, event_loc, relation, _ = data
                    input_word, input_entity, target = Variable(input_word).cuda(),Variable(input_entity).cuda(),Variable(target).cuda()
                
                    char_encode = []
                    for chars in input_char :
                        chars = char_cnn(Variable(chars).cuda())
                        char_encode.append(chars)
                    char_encode = torch.cat(char_encode,0) # L*N*conv_out_channel

                    hidden1,hidden2 = bilstm.initHidden()
                    bilstm.eval()
                    bilstm_output,hidden,entity_emb = bilstm((input_word,input_entity,char_encode),(hidden1.cuda(),hidden2.cuda()))
                    tr.eval()
                    tr_output,tr_emb,tr_index = tr(bilstm_output)

                    row = tr_index
                    previous_type = 'NONE'

                    event_loc_ = dict()
                    event_info = dict()
                    event_argu = dict()
                    
                    for k in range(0,len(row)) :
                        
                        index = int(row[k])
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

                    base_loc = base_loc+len(row)
                    
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
                            src = bilstm_output[src_begin:src_end+1]
                            src_event = tr_emb[src_begin:src_end+1]
                            dst = bilstm_output[dst_begin:dst_end+1]

                            src_name = event_index_r[int(tr_index[src_begin])].split('-')[0]
                            
                            event_flag = False
                            if event_loc_.has_key(dst_key):
                                dst_event = tr_emb[dst_begin:dst_end+1]
                                dst_name = event_index_r[int(tr_index[dst_begin])].split('-')[0]
                                event_flag = True
                            else :
                                dst_event = entity_emb[dst_begin:dst_end+1]
                                dst_name = entity_index_r[int(input_entity.data[0][dst_begin])].split('-')[0]
                            
                            reverse_flag = False
                            middle_flag = False
                            if  src_end+1 < dst_begin:
                                middle = bilstm_output[src_end+1:dst_begin]
                            elif dst_end < src_begin-1:
                                reverse_flag = True
                                middle = bilstm_output[dst_end+1:src_begin]
                            else : # adjacent or overlapped
                                middle_flag = True
                                middle = Variable(torch.zeros(1,1,128*2)).cuda() # L(=1)*N*2 self.hidden_size

                            rc.eval()
                            rc_output = rc(src_event,src,middle,dst,dst_event,reverse_flag,middle_flag)
                            '''
                            rc.eval()
                            rc_output = rc(bilstm_output,tr_emb,src_range,dst_range)
                            '''
                            row = rc_output.data
                            this_row = list(row[0])
                            index = this_row.index(max(this_row))
                            current_type = relation_index_r[index]
                            current_strength = this_row[index]
                            
                            if current_type != 'NONE' :
                                global_count = global_count + 1
                                if not relations.has_key(current_type) :
                                    relations[current_type] = {(dst_key,current_strength)}
                                else :
                                    relations[current_type].add((dst_key,current_strength))

                        event_argu[src_key] = dict(relations)

                key_list = list(event_loc_.keys())
                key_list.sort()
                
                t2e = dict()
                
                for t_id in key_list:
                    
                    info = event_info[t_id]
                    f.write(t_id+'\t'+info[0]+' '+str(info[1])+' '+str(info[2])+'\t'+info[3]+'\n')
                    
                    argus = illegal_argument_cut(info[0],event_argu[t_id])
                    #argus = missing_argument_fill(t_id,info[0],event_argu[t_id],e_loc)
                    
                    argus = getrate_all_combination(argus)
                    event_argu[t_id] = argus
                    
                    t2e[t_id] = e_index
                    if len(argus) == 0 :
                        e_index = e_index + 1
                    else :
                        e_index = e_index + len(argus)

                for t_id in key_list:
                    
                    argus = event_argu[t_id]
                    e_index_ = t2e[t_id]
                    e_id = 'E%d'%e_index_

                    if len(argus) == 0 :
                        f.write(e_id+'\t')
                        f.write(event_info[t_id][0]+':'+t_id)
                        f.write('\n')
                        continue

                    for argu in argus: 
                        f.write(e_id+'\t')
                        f.write(event_info[t_id][0]+':'+t_id)
                        
                        for item in argu:
                            argu_id = item[1]
                            if argu_id in t2e.keys() :
                                argu_id = 'E%d'%t2e[argu_id]
                            f.write(' '+item[0]+':'+argu_id)             
                        f.write('\n')

                        e_index_ = e_index_ + 1
                        e_id = 'E%d'%e_index_

            f.close()
    print global_count
