import os
import re
import csv
from nltk import sent_tokenize


def get_position_dict(file_path):
    
    src = open(file_path,'r')
    title = src.readline()
    abstract = src.readline()
    if abstract == '\n' : # patch for MLEE
        abstract = src.readline()
    src.close()

    title = title.replace('\n','')
    abstract = abstract.replace('\n','')
    sentence_list = [ title ]
    sentence_list.extend(sent_tokenize(abstract))
    
    position = 0
    position2word = dict()
    position2end = set()
    
    for i,sentence in enumerate(sentence_list) :
        # split with [ /-().,:] but reserve them
        words = re.split('([ -/().,:;])',sentence)
 
        for j,w in enumerate(words) :
            if w != ' ' and w != '' :
                position2word[position] = w

            if i != 0 :
                if j == len(words)-2 :
                    position2end.add(position)
            else :
                if j >= len(words)-4 and w == '.' :
                    position2end.add(position) # special for title
            position = position + len(w)
        
        # there is a \n behind the sentences in title and ' ' in abstract
        position = position + 1 # 1 for CG/PC, 2 for MLEE
        
    return position2word,position2end


def get_entity(file_path,position2word):
    
    src = open(file_path,'r')
    entity_notation = dict()
    entity_index = dict()

    line = src.readline()
    while line :
        words = re.split('[ \t]',line)
        index = words[0]
        type_ = words[1]
        start = int(words[2])
        end = int(words[3])
        
        i = 0
        while position2word[i+1][0] <= start : # some entity is a part of word
            i = i + 1
        start = position2word[i][0]
        
        entity_list = []
        for p2w in position2word:
            p = p2w[0]
            if ( p >= start and p < end ) :
                entity_list.append(p)
        
        if len(entity_list) == 1 :
            entity = entity_list[0]
            entity_notation[entity] = type_ + '-Unit'
        else :
            entity = entity_list[0]
            entity_notation[entity] = type_ + '-Begin'
            entity = entity_list[-1]
            entity_notation[entity] = type_ + '-Last'
            for i in range(1,len(entity_list)-1):
                entity = entity_list[i]
                entity_notation[entity] = type_ + '-Inside'

        for entity in entity_list :
            entity_index[entity] = index
            
        line = src.readline()
        
    src.close()
           
    return entity_notation,entity_index


def get_event(file_path,position2word):
    
    src = open(file_path,'r')
    event_notation = dict()
    event_index = dict()
    
    line = src.readline()
  
    while line :
        if line[0] != 'T' :
            line = src.readline()
            continue
      
        words = re.split('[ \t]',line)
        index = words[0]
        type_ = words[1]
        start = int(words[2])
        end = int(words[3])

        i = 0
        while position2word[i+1][0] <= start : # some event is a part of word
            i = i + 1
        start = position2word[i][0]

        event_list = []
        for p2w in position2word:
            p = p2w[0]
            if ( p >= start and p < end ) :
                event_list.append(p)

        if len(event_list) == 1 :
            event = event_list[0]
            event_notation[event] = type_ + '-Unit'
        else :
            event = event_list[0]
            event_notation[event] = type_ + '-Begin'
            event = event_list[-1]
            event_notation[event] = type_ + '-Last'
            for i in range(1,len(event_list)-1):
                event = event_list[i]
                event_notation[event] = type_ + '-Inside'

        for event in event_list :
            event_index[event] = index
            
        line = src.readline()
        
    src.close()
           
    return event_notation,event_index


if __name__ == '__main__':

    path_ = os.path.abspath('.')
    txt_path = path_ + '/txt' # 'E:/corpus/MLEE_test/train/txt'

    for fpath,_,files in os.walk(txt_path):
        for fl in files:
            print fl
        
            file_path = os.path.join(fpath,fl)
            position2word,position2end = get_position_dict(file_path)
           
            position2word = [(p,position2word[p]) for p in sorted(position2word.keys())] 

            file_path = file_path.replace('txt','a1')
            entity_notation,entity_index = get_entity(file_path,position2word)

            file_path = file_path.replace('a1','a2')
            event_notation,event_index = get_event(file_path,position2word) #dict(),dict()
            
            file_path = file_path.replace('/a2','/table')
            file_path = file_path.replace('.a2','.csv')

            csvfile = open(file_path,'wb')
            writer = csv.writer(csvfile)
            writer.writerow(['position','word',
                             'entity_index','entity_notation',
                             'event_index','event_notation',
                             'is_end'])
            csvfile.close()

            csvfile = open(file_path,'ab+')
            writer = csv.writer(csvfile)
            
            for p2w in position2word :

                p = p2w[0]
                w = p2w[1]
                
                if entity_index.has_key(p) :
                    e_i = entity_index[p]
                    e_n = entity_notation[p]
                else :
                    e_i,e_n = '',''
                    
                if event_index.has_key(p) :
                    ev_i = event_index[p]
                    ev_n = event_notation[p]
                else :
                    ev_i,ev_n = '',''

                if p in position2end :
                    flag = True
                else :
                    flag = False
                    
                writer.writerow([ p, w, e_i, e_n, ev_i, ev_n, flag])
                
            csvfile.close()
