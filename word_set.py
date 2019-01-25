import os
import re
import torch
import cPickle
import numpy as np
import torch.utils.data as data
from pandas.io.parsers import read_csv


def get_all_words(folder):
    
    all_words = []
    all_event = []

    for root, _, fnames in os.walk(folder):
        for fname in fnames:
            
            path = os.path.join(root, fname)
            df = read_csv(path)
            
            for i in range(0,len(df)):
                #if not df['entity_notation'][i] is np.nan :
                #    word = df['entity_notation'][i]
                #else :
                word = df['word'][i]
		all_words.append(str(word))
		if not df['event_notation'][i] is np.nan :
                    all_event.append(df['event_notation'][i])
                else :
                    all_event.append('NONE')

    return all_words,all_event


def assign_char_index(all_words): 
    
    char_index = {}

    for word in all_words:
        for char in word :
            if char not in char_index.keys():
                char_index[char] = len(char_index)

    char_index['OTHER'] = len(char_index)

    path_ = os.path.abspath('.')
    f = file(path_+'/char_index', 'w')
    cPickle.dump(char_index, f)

    return char_index


def assign_event_index(all_events): 
    
    event_index = {}

    for event in all_events:
        if event not in event_index.keys():
            event_index[event] = len(event_index)

    event_index['OTHER'] = len(event_index)

    path_ = os.path.abspath('.')
    f = file(path_+'/event_index', 'w')
    cPickle.dump(event_index, f)

    return event_index


def word_to_onehot(keys,key_index): # map a key list into one-hot index list
    
    length = len(keys)
    tensor = torch.LongTensor(1,length).zero_() # N*L
    for i,k in enumerate(keys):
        if key_index.has_key(k) :
            tensor[0][i] = key_index[k]
        else :
            tensor[0][i] = key_index['OTHER']

    return tensor


def event_to_onehot(key,key_index): # map a key into one-hot index
    
    tensor = torch.LongTensor(1).zero_()
    if key_index.has_key(key) :
        tensor[0] = key_index[key]
    else :
        tensor[0] = key_index['NONE']

    return tensor


class Word_Set(data.Dataset):

    def __init__(self, file_dir, new_dict):

        words,events = get_all_words(file_dir)

        if new_dict :
            self.char_index = assign_char_index(words)
            self.event_index = assign_event_index(events)
        else :
            path_ = os.path.abspath('.')
            f = file(path_+'/char_index', 'r')
            self.char_index = cPickle.load(f)
            f = file(path_+'/event_index', 'r')
            self.event_index = cPickle.load(f)

        self.words = words
	self.events = events

    def get_char_dim(self):
        return len(self.char_index)

    def get_event_dim(self):
        return len(self.event_index)

    def __getitem__(self, index):
        word = self.words[index]
	event = self.events[index]
        input_ = word_to_onehot(word,self.char_index)
        target = event_to_onehot(event,self.event_index)
        return input_, target

    def __len__(self):
        return len(self.words)
