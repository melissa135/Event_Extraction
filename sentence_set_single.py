import os
import re
import torch
import cPickle
import numpy as np
import torch.utils.data as data
from pandas.io.parsers import read_csv
from word_set import word_to_onehot,event_to_onehot
from sentence_set import char_to_onehot


def get_all_sentences(fname):

    all_sentences = []
    all_entitys = []
    all_events = []
    all_word_loc = []
    all_entity_loc = []
    all_event_loc = []
    all_relations = []
    all_event_paras = []

    df = read_csv(fname)

    fname_a2 = fname.replace('table','table_a2')
    df_a2 = read_csv(fname_a2)
    e2t = dict() # convert Exx to Txx
    for j in range(0,len(df_a2)):
        e_id = df_a2['event_id'][j]
        t_id = df_a2['src'][j]
        e2t[e_id] = t_id

    sentence = []
    entitys = []
    events = []
    word_loc = []
    entity_loc = dict()
    event_loc = dict()
    event_paras = dict() # map event to its parameter set
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

        position = (df['position'][i],df['position'][i]+len(df['word'][i]))
        word_loc.append(position)

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

                if rlt is np.nan :
                    if src in id_set :
                        event_paras[src] = set()
                    continue

                if rlt == 'CSite' :
                    rlt = 'Site'
                if rlt[-1] >= '2' and rlt[-1] <= '9' :
                    rlt = rlt[:len(rlt)-1]
                if isinstance(dst,basestring) and dst[0] == 'E' :
                    dst = e2t[dst]

                if (src in id_set) and (dst in id_set) :
                    key = (src,dst)
                    relation[key] = rlt
                    if not event_paras.has_key(src):
                        event_paras[src] = set()
                        event_paras[src].add((e_id,rlt,dst))
                    else :
                        event_paras[src].add((e_id,rlt,dst))

            for src in event_loc.keys() : # all invalid relations in a single sentence
                for dst in entity_loc.keys() : # event to entity
                    if src == dst :
                        continue
                    if not (src,dst) in relation.keys() :
                        relation[(src,dst)] = 'NONE'

            for src in event_loc.keys() : # all invalid relations in a single sentence
                for dst in event_loc.keys() : # event to event
                    if src == dst :
                        continue
                    if not (src,dst) in relation.keys() :
                        relation[(src,dst)] = 'NONE'

            all_sentences.append(sentence[:])
            all_entitys.append(entitys[:])
            all_events.append(events[:])
            all_word_loc.append(word_loc[:])
            all_entity_loc.append(entity_loc)
            all_event_loc.append(event_loc)
            all_relations.append(relation)
            all_event_paras.append(event_paras)

            sentence = []
            entitys = []
            events = []
            word_loc = []
            entity_loc = dict()
            event_loc = dict()
            event_paras = dict()
            start_index = i+1

    return all_sentences,all_entitys,all_events,all_word_loc,all_entity_loc,all_event_loc,all_relations,all_event_paras


class Sentence_Set_Single(data.Dataset):

    def __init__(self, filename):

        sentences,entitys,events,word_loc,entity_loc,event_loc,relations, event_paras = get_all_sentences(filename)

        path_ = os.path.abspath('.')
        f = file(path_+'/char_index', 'r')
        self.char_index = cPickle.load(f)
        f = file(path_+'/entity_index', 'r')
        self.entity_index = cPickle.load(f)
        f = file(path_+'/event_index', 'r')
        self.event_index = cPickle.load(f)
        f = file(path_+'/word_index', 'r') ###
        self.word_index = cPickle.load(f)
        f = file(path_+'/relation_index', 'r')
        self.relation_index = cPickle.load(f)

        self.sentences = sentences
        self.entitys = entitys
        self.events = events
        self.word_loc = word_loc
        self.entity_loc = entity_loc
        self.event_loc = event_loc
        self.event_paras = event_paras

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

        input_word = word_to_onehot(sentence,self.word_index) # N(=1)*L
        input_entity = word_to_onehot(entity,self.entity_index)
        input_char = char_to_onehot(sentence_char,self.char_index) # L*( N(=1)*Char_L ) list
        target = word_to_onehot(event,self.event_index) # N(=1)*L

        return input_word, input_entity, input_char, target, entity_loc, event_loc, relation, event_para

    def __len__(self):
        return len(self.sentences)