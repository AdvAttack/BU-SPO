
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle

import os
#import nltk
import re
from collections import Counter


import data_utils_yahoo
import glove_utils

Yahoo_PATH = 'yahoo_answers_csv'
MAX_VOCAB_SIZE = 50000
GLOVE_PATH = 'glove.840B.300d.txt'

if not os.path.exists('aux_files'):
	os.mkdir('aux_files')
yahoo_dataset = data_utils_yahoo.YahooDataset(path=Yahoo_PATH, max_vocab_size=MAX_VOCAB_SIZE)

# save the dataset
with open(('aux_files/small_dataset_%d.pkl' %(MAX_VOCAB_SIZE)), 'wb') as f:
    pickle.dump(yahoo_dataset, f)

# create the glove embeddings matrix (used by the classification model)
glove_model = glove_utils.loadGloveModel(GLOVE_PATH)
glove_embeddings, _ = glove_utils.create_embeddings_matrix(glove_model, yahoo_dataset.dict, yahoo_dataset.full_dict)
# save the glove_embeddings matrix
np.save('aux_files/small_embeddings_glove_%d.npy' %(MAX_VOCAB_SIZE), glove_embeddings)




print('All done')
