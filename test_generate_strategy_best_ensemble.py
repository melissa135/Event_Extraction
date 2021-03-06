from pandas.io.parsers import read_csv
from dataloader_modified import DataLoader
from define_net import Char_CNN_encode,BiLSTM,Trigger_Recognition,Relation_Classification
from sentence_set_single import Sentence_Set_Single
from define_net_event_evaluation import *
from generate_utils import *


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

        
def Combination_Strategy(src_key,event_support,relation_support,candidate_event,alpha,beta,gamma):
    
    best_penalty = event_support[src_key]*beta
    best_combination = []
    best_range = 0
    related_relation = dict()
    
    for key in relation_support.keys() :
        if key[0] == src_key :
            related_relation[key] = relation_support[key]
            best_penalty = best_penalty + related_relation[key]*gamma
    additional_penalty = best_penalty
                        
    for j in range(0,len(candidate_event)) :
        best_penalty_this_round = 9999.9 # infinite max
        penalty = 0.0
                                    
        for event in candidate_event :
            if event.choosed == True :
                penalty = penalty + max(1.0-event.support*alpha,0.0) #-event.support*0.4
                                    
        for event in candidate_event :
            if event.choosed == False :
                penalty_ = penalty + max(1.0-event.support*alpha,0.0) # -event.support*0.4
                para_nodes = event.get_all_para_nodes()
                for key in related_relation.keys() :
                    if not key[1] in para_nodes :
                        penalty_ = penalty_ + related_relation[key]*gamma
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
    additional_penalty = additional_penalty - best_penalty
    
    return best_combination, additional_penalty  


def resolution_recursive(t2e, event_info_paras):
    
    e2e = dict()
    for t_id in t2e.keys():
        info = event_info[t_id]
                     
        for e_id in t2e[t_id] :
            continue_flag = False
            recursive_flag = False
            argus = event_info_paras[e_id]
            e2e[e_id] = set()
                            
            for argu in argus :
                argu_id = argu[1]
                if len(argu_id) > 4 and (int(argu_id[1:])>=START_T) and not t2e.has_key(argu_id):
                    continue_flag = True  # refer to undefined events
                            
            if continue_flag :
                continue
                         
            for argu in argus :
                argu_id = argu[1]
                if argu_id == 'None' :
                    continue
                if argu_id in t2e.keys() :
                    argu_id = t2e[argu_id][0] #
                    e2e[e_id].add(argu_id)
                    if e_id in e2e.get(argu_id,set()) :
                        recursive_flag = True
                elif argu_id[0] == 'E' :
                    e2e[e_id].add(argu_id)
                    if e_id in e2e.get(argu_id,set()) :
                        recursive_flag = True
                                        
            if recursive_flag :
                t2e[t_id].remove(e_id)
                event_info_paras.pop(e_id)

    return t2e, event_info_paras


def delete_unstable(t2e,event_info_paras):
                            
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
                if (len(argu_id) == 5) and (int(argu_id[1:])>=START_T) and (argu_id not in t_key_list) :
                    delete = True
                if argu_id in t2e.keys() :
                    if len(t2e[argu_id]) != 0 :
                        argu_id = t2e[argu_id][0]
                    else :
                        delete = True
                if (argu_id[0] == 'E') and (argu_id not in e_key_list):
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
                                
    return t2e, event_info_paras


if __name__ == '__main__':

    path_ = os.path.abspath('.')
    folder = path_+'/table_test/'
    dst_folder = path_+'/a2_result/'

    ensemble_epoch = 1
    ensemble_number = 1
    bias_tr,bias_rc = 2.0, 2.0
    none_resize_tr,none_resize_rc,none_resize_m = 2.0, 2.0, 3.0
    alpha, beta, gamma = 0.5, 0.25, 0.125

    have_load = False
    START_T = 2000
    ensemble = ensemble_epoch*ensemble_number
 
    for root, _, fnames in os.walk(folder):
        for fname in fnames:
            if fname.split('.')[1] != 'csv' :
                continue
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

                char_cnn_list = [ Char_CNN_encode(char_dim) for k in range(0,ensemble) ]
                bilstm_list = [ BiLSTM(word_dim,entity_dim) for k in range(0,ensemble) ]
                tr_list = [ Trigger_Recognition(event_dim) for k in range(0,ensemble) ]
                rc_list = [ Relation_Classification(relation_dim) for k in range(0,ensemble) ]
                ee_list = [ Event_Evaluation() for k in range(0,ensemble) ]

                epoch = 20
                base = 0
                for k in range(0,ensemble_epoch):
                    for l in range(0,ensemble_number):
                        char_cnn_list[k*ensemble_number+l].load_state_dict(torch.load(path_+'/network/char_cnn_%d_%d.pth'%(epoch+10*k,base+l+2)))
                        bilstm_list[k*ensemble_number+l].load_state_dict(torch.load(path_+'/network/bilstm_%d_%d.pth'%(epoch+10*k,base+l)))
                        tr_list[k*ensemble_number+l].load_state_dict(torch.load(path_+'/network/tr_%d_%d.pth'%(epoch+10*k,base+l)))
                        rc_list[k*ensemble_number+l].load_state_dict(torch.load(path_+'/network/rc_%d_%d.pth'%(epoch+10*k,base+l)))
                        ee_list[k*ensemble_number+l].load_state_dict(torch.load(path_+'/network/ee_%d_%d.pth'%(epoch+10*k,base+l)))

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

            t_index = START_T # Entity number is unimportant, begin with a large numberto avoid overlapping with entity numbers
            e_index = 1
            m_index = 0
            base_loc = 0

            result_dir = dst_folder + fname.replace('.csv','.a2')
            f = file(result_dir,'w')
            
            for i,batch in enumerate(testloader,0):
                    
                for data in batch: # due to we have modified the defination of batch, the batch here is a list

                    input_word, input_entity, input_char, target, entity_loc, event_loc, relation, _ = data
                    input_word, input_entity, target = Variable(input_word),Variable(input_entity),Variable(target)

                    bilstm_output_list = []
                    entity_emb_list = []
                    tr_emb_list = []
                    tr_index = []
                    
                    for k in range(0,ensemble):
                        char_cnn = char_cnn_list[k]
                        bilstm = bilstm_list[k]
                        tr = tr_list[k]
                        
                        char_encode = []
                        for chars in input_char :
                            chars = char_cnn(Variable(chars))
                            char_encode.append(chars)
                        char_encode = torch.cat(char_encode,0) # L*N*conv_out_channel

                        hidden1,hidden2 = bilstm.initHidden()
                        bilstm.eval()
                        bilstm_output,hidden,entity_emb = bilstm((input_word,input_entity,char_encode),(hidden1,hidden2))
                        bilstm_output_list.append(bilstm_output)
                        entity_emb_list.append(entity_emb)
                        '''
                        tr.eval()
                        tr_output,tr_emb,_ = tr(bilstm_output)
                        tr_emb_list.append(tr_emb)

                        row = tr_output.data

                        for x in range(0,row.size()[0]) :
                            row[x][event_index['NONE']] = row[x][event_index['NONE']]*none_resize_tr
                        
                        if k == 0 :
                            row_sum = row[:]
                        else :
                            for x in range(0,row.size()[0]) :
                                for y in range(0,row.size()[1]) :
                                    row_sum[x][y] = row_sum[x][y] + row[x][y]

                    row_sum_new = []
                    for l in range(0,row_sum.size()[0]):
                        row_sum_new.append(list(row_sum[l]))
                        for j in range(0,len(row_sum_new[l])) :
                            row_sum_new[l][j] = float(row_sum_new[l][j].data)
                        row_sum_new[l][event_index['NONE']] = row_sum_new[l][event_index['NONE']] - bias_tr
                        index = row_sum_new[l].index(max(row_sum_new[l]))
                        tr_index.append(index)

                        for k in range(0,ensemble):
                            tr = tr_list[k]
                            vector = tr.get_event_embedding(index)
                            tr_emb_list[k][l] = vector
                    '''
                    row_sum_new = []
                    length = bilstm_output_list[0].size()[0]
                    embedding_size = tr_list[0].embedding_size

                    for l in range(0,length):
                        for k in range(0,ensemble):
                            if l == 0 :
                                last_index = event_index['NONE']
                                tr_emb_list.append(torch.zeros(length,1,embedding_size))
                            else :
                                last_index = index
                            tr = tr_list[k]
                            bilstm_output = bilstm_output_list[k]
                            tr.eval()
                            tr_output,tr_emb,_ = tr(bilstm_output[l:l+1],event_index=last_index)
                            if k == 0 :
                                row_sum_new.append(tr_output.view(-1).data)
                            else :
                                for j in range(0,tr_output.size()[1]):
                                    row_sum_new[l][j] = row_sum_new[l][j] + tr_output.view(-1).data[j]

                        row_sum_new[l][event_index['NONE']] = row_sum_new[l][event_index['NONE']]*none_resize_tr
                        row_sum_new[l][event_index['NONE']] = row_sum_new[l][event_index['NONE']] - bias_tr

                        row = list(row_sum_new[l])
                        index = row.index(max(row))
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

                    if index != event_index['NONE'] :
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
                            support = support + float(row_sum_new[j][index])/ensemble-\
                                      float(row_sum_new[j][event_index['NONE']])/ensemble
                        support = support/(end-start+1)
                        event_support[key] = support - bias_tr

                    base_loc = base_loc+len(row_sum_new)
                    
                    e_loc = dict(entity_loc.items()+event_loc_.items()) # event refer to net output

                    event_para_ = dict()
                    src_type = dict()
                    dst_type = dict()
                    relation_support = dict()
                        
                    for src_key in event_loc_.keys() :
                       
                        for dst_key in e_loc.keys() :
                            if src_key == dst_key :
                                continue
                            
                            src_begin,src_end = event_loc_[src_key]
                            dst_begin,dst_end = e_loc[dst_key]

                            src_name = event_index_r[int(tr_index[src_begin])].split('-')[0]
                            src_type[src_key] = src_name

                            if event_loc_.has_key(dst_key):
                                dst_name = event_index_r[int(tr_index[dst_begin])].split('-')[0]
                                event_flag = True
                            else :
                                dst_name = entity_index_r[int(input_entity.data[0][dst_begin])].split('-')[0]
                                event_flag = False
                            dst_type[dst_key] = dst_name

                            reverse_flag = False
                            if src_begin > dst_begin :
                                reverse_flag = True

                            for k in range(0,ensemble):

                                bilstm_output = bilstm_output_list[k]
                                entity_emb = entity_emb_list[k]
                                tr_emb = tr_emb_list[k]
                                '''
                                src = bilstm_output[src_begin:src_end+1]
                                src_event = tr_emb[src_begin:src_end+1]
                                dst = bilstm_output[dst_begin:dst_end+1]
                                
                                if event_flag:
                                    dst_event = tr_emb[dst_begin:dst_end+1]
                                else :
                                    dst_event = entity_emb[dst_begin:dst_end+1]
                                '''
                                src = torch.cat((bilstm_output[src_begin:src_end+1],\
                                                 tr_emb[src_begin:src_end+1],entity_emb[src_begin:src_end+1]),2)
                                dst = torch.cat((bilstm_output[dst_begin:dst_end+1],\
                                                 tr_emb[dst_begin:dst_end+1],entity_emb[dst_begin:dst_end+1]),2)
                                
                                middle_flag = False
                                if  src_end+1 < dst_begin:
                                    #middle = bilstm_output[src_end+1:dst_begin]
                                    middle = torch.cat((bilstm_output[src_end+1:dst_begin],\
                                                        tr_emb[src_end+1:dst_begin],entity_emb[src_end+1:dst_begin]),2)
                                elif dst_end < src_begin-1:
                                    #middle = bilstm_output[dst_end+1:src_begin]
                                    middle = torch.cat((bilstm_output[dst_end+1:src_begin],\
                                                        tr_emb[dst_end+1:src_begin],entity_emb[dst_end+1:src_begin]),2)
                                else : # adjacent or overlapped
                                    #middle_flag = True
                                    #middle = Variable(torch.zeros(1,1,128*2)) # L(=1)*N*2 self.hidden_size
                                    middle = torch.FloatTensor(0)
                                    
                                rc = rc_list[k]
                                
                                rc.eval()
                                #rc_output = rc(src_event,src,middle,dst,dst_event,reverse_flag,middle_flag)
                                rc_output = rc(src,middle,dst,reverse_flag)

                                row = rc_output.data
                                row[0][relation_index['NONE']] = row[0][relation_index['NONE']]*none_resize_tr
                                if k == 0 :
                                    row_sum = row[:]
                                else :
                                    for x in range(0,row.size()[0]) :
                                        for y in range(0,row.size()[1]) :
                                            row_sum[x][y] = row_sum[x][y] + row[x][y]
                                
                            this_row = list(row_sum[0])
                            index = this_row.index(max(this_row))
                            current_type = relation_index_r[index]

                            # three useless statements, caused 0.4% decent of f1score
                            # but here it is useful to prevent event recursion!
                            if event_flag and not src_name in {'Regulation','Positive_regulation','Negative_regulation','Planned_process'} :
                                continue
                            if current_type == 'Site' and dst_name != 'Gene_or_gene_product':
                                continue
                            if event_flag and current_type == 'Instrument' and src_name == 'Planned_process':
                                continue
                            
                            if not event_para_.has_key(src_key) : # key point, keep the event with no relations
                                event_para_[src_key] = set()
                            
                            if current_type != 'NONE' and current_type != 'OTHER' :
                                support = (this_row[index] - this_row[relation_index['NONE']])/ensemble
                                relation_support[(src_key,dst_key)] = support - bias_rc
                                event_para_[src_key].add((current_type,dst_key))

                    for src_key in event_loc_.keys(): # add the isolate event (with no paras)
                        if not event_para_.has_key(src_key) :
                            event_para_[src_key] = set()
                        if not src_type.has_key(src_key) :
                            src_begin = event_loc_[src_key][0]
                            src_type[src_key] = event_index_r[int(tr_index[src_begin])].split('-')[0]

                    dst_type['None'] = 'None'
                    e_loc['None'] = (0,-1)
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
                        #range_list = []
                        range_lists,para_sets = generate_all_combination(src_type[src_key],event_para_[src_key],e_loc)

                        for k in range(0,ensemble):

                            bilstm_output = bilstm_output_list[k]
                            entity_emb = entity_emb_list[k]
                            tr_emb = tr_emb_list[k]

                            event_evaluation = ee_list[k]

                            support_list = []
                            m_list = []
                            #dst_set = []

                            for j in range(0,len(range_lists)):
                                range_list = range_lists[j]
                                para_set = para_sets[j]
                                hidden = event_evaluation.initHidden()
                                r,m = event_evaluation(bilstm_output,tr_emb,entity_emb,s_r,range_list,hidden)

                                support_list.append(float(r.data[0][1] - r.data[0][0]))
                                m_list.append(list(m.data[0]))

                            if k == 0 :
                                support_sum = support_list[:]
                                m_list_sum = m_list[:]
                            else :
                                support_sum = [ support_sum[j] + support_list[j] for j in range(0,len(support_sum)) ]
                                m_list_sum = [ [ m_list_sum[j][jj]+m_list[j][jj] for jj in range(0,3) ] for j in range(0,len(m_list_sum)) ]

                        candidate_event = []
                        for j in range(0,len(support_sum)) :
                            m_list_sum[j][0] = m_list_sum[j][0] * none_resize_m
                            m_index_ = m_list_sum[j].index(max(m_list_sum[j]))
                            event = Event(src_key,para_sets[j],support_sum[j]/ensemble,m_index_)
                            candidate_event.append(event)

                        best_combination, additional_penalty = Combination_Strategy(src_key,event_support,relation_support,candidate_event,alpha,beta,gamma)
                        
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
                                    event_support[dst_key] = event_support[dst_key] + additional_penalty # has the accumulate effect
                                # result in a sightly decrease, but have to reserve

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
                        # begin
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
                                new_argus = argus[:]
                                new_argus.append((argu[0],argu_ids[n]))
                                e_id_ = 'E%d'%e_index
                                e_index = e_index + 1
                                t2e_t_id.append(e_id_)
                                e2m[e_id_] = e2m[e_id]
                                event_info_paras[e_id_] = new_argus

                            argus.append((argu[0],argu_ids[0]))
                            # end : an effective patch result in a 0.35% improvemnt
                        t2e[t_id] = t2e_t_id

                    t2e, event_info_paras = resolution_recursive(t2e,event_info_paras)
                    t2e, event_info_paras = delete_unstable(t2e,event_info_paras)
     
                    for t_id in t2e.keys():
                        info = event_info[t_id]
                     
                        for e_id in t2e[t_id] :
                            continue_flag = False
                            argus = event_info_paras[e_id]
                            
                            for argu in argus : # refer to undefined events
                                argu_id = argu[1]
                                if (len(argu_id) == 5) and (int(argu_id[1:])>START_T) and not t2e.has_key(argu_id):
                                    continue_flag = True
                            
                            if continue_flag :
                                print argus
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
                            
                            if e2m[e_id] != 'None' : # only for CG and PC
                                m_index = m_index + 1
                                f.write('M%d'%m_index+'\t'+e2m[e_id]+' '+e_id+'\n')
                            
            f.close()
