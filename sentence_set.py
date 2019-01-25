import os
import re
import torch
import cPickle
import numpy as np
import torch.utils.data as data
from pandas.io.parsers import read_csv
from word_set import word_to_onehot,event_to_onehot


def get_all_sentences(folder):

    all_sentences = []
    all_entitys = []
    all_events = []
    all_entity_loc = []
    all_event_loc = []
    all_relations = []
    all_event_paras = []
    all_modifications = []
    all_fnames = []

    for root, _, fnames in os.walk(folder):
        for fname in fnames:

            path = os.path.join(root, fname)
            df = read_csv(path)

            path_a2 = path.replace('table','table_a2')
            df_a2 = read_csv(path_a2)
            e2t = dict() # convert Exx to Txx

            for j in range(0,len(df_a2)):
                e_id = df_a2['event_id'][j]
                src = df_a2['src'][j]
                e2t[e_id] = src

            sentence = []
            entitys = []
            events = []
            entity_loc = dict()
            event_loc = dict()
            event_paras = dict() # map event to its parameter set
            event_modifications = dict()
            eid_set = set()
            start_index = 0

            for i in range(0,len(df)):

                word = df['word'][i]
                sentence.append(str(word))

                if not df['entity_notation'][i] is np.nan :
                    entitys.append(df['entity_notation'][i])
                    key = df['entity_index'][i]
                    if not key in entity_loc.keys() :
                        entity_loc[key] = (i-start_index,i-start_index) # entity location:(start,end)
                    else :
                        begin = entity_loc[key][0]
                        entity_loc[key] = (begin,i-start_index) # update the end location
                else :
                    entitys.append('NONE')

                if not df['event_notation'][i] is np.nan :
                    events.append(df['event_notation'][i])
                    key = df['event_index'][i]
                    if not key in event_loc.keys() :
                        event_loc[key] = (i-start_index,i-start_index) # event location:(start,end)
                    else :
                        begin = event_loc[key][0]
                        event_loc[key] = (begin,i-start_index) # update the end location
                else :
                    events.append('NONE')

                if df['is_end'][i] == True :

                    relation = dict() # (src,dst):relation_type
                    id_set = entity_loc.keys() + event_loc.keys()

                    for j in range(0,len(df_a2)):  # all valid relations in a single sentence
                        src = df_a2['src'][j]
                        dst = df_a2['dst'][j]
                        rlt = df_a2['relation'][j]
                        e_id = df_a2['event_id'][j]

                        if (rlt is np.nan) :
                            if src in id_set :
                                event_paras[src] = set()
                            continue

                        try : # a simplification, i.e. theme3/theme4/.. -> theme2
                            if rlt[-1] >= '3' and rlt[-1] <= '9' :
                                rlt = rlt[:len(rlt)-1]
                                rlt = rlt + '2'
                        except Exception as e: # same to prevent rlt == np.nan, patch for MLEE
                            if src in id_set :
                                event_paras[src] = set()
                            continue

                        if rlt == 'CSite' :
                            rlt = 'Site'
                        
                        if isinstance(dst,basestring) and dst[0] == 'E' :
                            dst = e2t[dst]

                        if (src in id_set) and (dst in id_set) :
                            key = (src,dst)
                            relation[key] = rlt
                            eid_set.add(e_id)
                            if not event_paras.has_key(src):
                                event_paras[src] = set()
                                event_paras[src].add((e_id,rlt,dst))
                            else :
                                event_paras[src].add((e_id,rlt,dst))

                    for src in event_loc.keys() : # all invalid relations in a single sentence
                        for dst in event_loc.keys() : # event to entity
                            if src == dst :
                                continue
                            if not (src,dst) in relation.keys() :
                                relation[(src,dst)] = 'NONE'

                    for src in event_loc.keys() : # all invalid relations in a single sentence
                        for dst in entity_loc.keys() : # event to event
                            if src == dst :
                                continue
                            if not (src,dst) in relation.keys() :
                                relation[(src,dst)] = 'NONE'

                    path_a2m = path.replace('table_','table_a2_m')
                    df_a2m = read_csv(path_a2m)

                    for j in range(0,len(df_a2m)):
                        dst = df_a2m['dst'][j]
                        mtype = df_a2m['modification_type'][j]
                        if dst in eid_set :
                            event_modifications[dst] = mtype

                    eid_set = set()
                    all_sentences.append(sentence[:])
                    all_entitys.append(entitys[:])
                    all_events.append(events[:])
                    all_entity_loc.append(entity_loc)
                    all_event_loc.append(event_loc)
                    all_relations.append(relation)
                    all_event_paras.append(event_paras)
                    all_modifications.append(event_modifications)
                    all_fnames.append(fname)
                    '''
                    # for some rare instance, the event and entity can be overlapped
                    for key1 in event_loc :
                        for key2 in entity_loc :
                            begin1,end1 = event_loc[key1]
                            begin2,end2 = entity_loc[key2]
                            if begin1 < begin2 and end1 > begin2 :
                                print key1,key2,fname
                            if begin2 < begin1 and end2 > begin1 :
                                print key1,key2,fname
                    '''
                    sentence = []
                    entitys = []
                    events = []
                    entity_loc = dict()
                    event_loc = dict()
                    event_paras = dict()
                    start_index = i+1

    return all_sentences,all_entitys,all_events,all_entity_loc,all_event_loc, \
           all_relations,all_event_paras,all_modifications,all_fnames


def assign_word_index(all_sentences):

    word_index = {}

    for sentence in all_sentences:
        for word in sentence:
            if word not in word_index.keys():
                word_index[word] = len(word_index)

    word_index['OTHER'] = len(word_index)

    path_ = os.path.abspath('.')
    f = file(path_+'/word_index', 'w')
    cPickle.dump(word_index, f)

    return word_index


def assign_entity_index(all_entitys):

    entity_index = {}

    for entitys in all_entitys:
        for entity in entitys:
            if entity not in entity_index.keys():
                entity_index[entity] = len(entity_index)

    entity_index['OTHER'] = len(entity_index)

    path_ = os.path.abspath('.')
    f = file(path_+'/entity_index', 'w')
    cPickle.dump(entity_index, f)

    return entity_index


def assign_relation_index(all_relations):

    relation_index = {}

    for relation in all_relations:
        for rlt in relation.values():
            if rlt not in relation_index.keys():
                relation_index[rlt] = len(relation_index)

    #relation_index['OTHER'] = len(relation_index)

    path_ = os.path.abspath('.')
    f = file(path_+'/relation_index', 'w')
    cPickle.dump(relation_index, f)

    return relation_index


def char_to_onehot(key_list,key_index): # map a list of char list into one-hot index

    tensor_list = []

    for keys in key_list:
        length = len(keys)
        tensor = torch.LongTensor(1,length).zero_() # N*L
        for i,k in enumerate(keys):
            if key_index.has_key(k) :
                tensor[0][i] = key_index[k]
            else :
                tensor[0][i] = key_index['OTHER']
        tensor_list.append(tensor)

    return tensor_list


class Sentence_Set(data.Dataset):

    def __init__(self, file_dir, new_dict=True):

        sentences,entitys,events,entity_loc,event_loc,relations,event_paras,modifications,fnames = get_all_sentences(file_dir)

        path_ = os.path.abspath('.')
        f = file(path_+'/char_index', 'r')
        self.char_index = cPickle.load(f)
        f = file(path_+'/event_index', 'r')
        self.event_index = cPickle.load(f)

        if new_dict :
            self.word_index = assign_word_index(sentences)
            self.entity_index = assign_entity_index(entitys)
            self.relation_index = assign_relation_index(relations)
        else :
            f = file(path_+'/word_index', 'r') ###
            self.word_index = cPickle.load(f)
            f = file(path_+'/entity_index', 'r')
            self.entity_index = cPickle.load(f)
            f = file(path_+'/relation_index', 'r')
            self.relation_index = cPickle.load(f)

        self.sentences = sentences
        self.entitys = entitys
        self.events = events
        self.entity_loc = entity_loc
        self.event_loc = event_loc
        self.event_paras = event_paras
        self.modifications = modifications
        self.fnames = fnames

        for i in range(0,len(relations)) :
            relation = relations[i]
            for key in relation.keys():
                value = relation[key]
                relation[key] = event_to_onehot(value,self.relation_index)
        self.relations = relations

    def get_char_dim(self):
        return len(self.char_index)

    def get_entity_dim(self):
        return len(self.entity_index)

    def get_event_dim(self):
        return len(self.event_index)

    def get_word_dim(self):
        return len(self.word_index)

    def get_relation_dim(self):
        return len(self.relation_index)

    def __getitem__(self, index):
        sentence = self.sentences[index]
        sentence_char = self.sentences[index]
        entity = self.entitys[index]
        event = self.events[index]
        relation = self.relations[index]
        entity_loc = self.entity_loc[index]
        event_loc = self.event_loc[index]
        event_para = self.event_paras[index]
        modification = self.modifications[index]
        fname = self.fnames[index]

        input_word = word_to_onehot(sentence,self.word_index) # N(=1)*L
        input_entity = word_to_onehot(entity,self.entity_index)
        input_char = char_to_onehot(sentence_char,self.char_index) # L*( N(=1)*Char_L ) list
        target = word_to_onehot(event,self.event_index) # N(=1)*L

        return input_word, input_entity, input_char, target, entity_loc, \
               event_loc, relation, event_para, modification, fname

    def __len__(self):
        return len(self.sentences)
