
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle

import os
#import nltk
import re
from collections import Counter


import data_utils_agnews
import glove_utils

AGNews_PATH = 'ag_news_csv'
MAX_VOCAB_SIZE = 50000
GLOVE_PATH = 'glove.840B.300d.txt'

if not os.path.exists('aux_files'):
	os.mkdir('aux_files')
agnews_dataset = data_utils_agnews.AGNewsDataset(path=AGNews_PATH, max_vocab_size=MAX_VOCAB_SIZE)

# save the dataset
with open(('aux_files/dataset_%d.pkl' %(MAX_VOCAB_SIZE)), 'wb') as f:
    pickle.dump(agnews_dataset, f)

# create the glove embeddings matrix (used by the classification model)
glove_model = glove_utils.loadGloveModel(GLOVE_PATH)
glove_embeddings, _ = glove_utils.create_embeddings_matrix(glove_model, agnews_dataset.dict, agnews_dataset.full_dict)
# save the glove_embeddings matrix
np.save('aux_files/embeddings_glove_%d.npy' %(MAX_VOCAB_SIZE), glove_embeddings)




print('All done')
