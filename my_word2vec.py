import os
import gensim
from gensim.models import Word2Vec
from pandas.io.parsers import read_csv

def read_files(folder):
    
    all_paper = []

    for root, _, fnames in os.walk(folder):
        for fname in fnames:
            
            path = os.path.join(root, fname)
            df = read_csv(path)
            content = []
            
            for i in range(0,len(df)):
		word = df['word'][i]
		content.append(word)
                
	    all_paper.append(content)

    return all_paper

##########################################
# Here is only a small corpus for demo.
# Using larger literatures for training is better.
##########################################

path_ = os.path.abspath('.')
paper_root = path_+'/table/'
sentences = read_files(paper_root)
#paper_root = path_+'/table_test/'
#sentences_ = read_files(paper_root)
#sentences.extend(sentences_)

model = Word2Vec(sentences, size=128, min_count=1, iter=32)
print len(sentences)
print len(model.wv.vocab)

model.save(path_+'/network/myword2vec')
