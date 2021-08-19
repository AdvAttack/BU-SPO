import os
# import nltk
import re
from collections import Counter
# from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
import csv
import pickle as pickle
import numpy as np


def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('', text)


class YahooDataset(object):
    def __init__(self, path='yahoo_answers_csv', max_vocab_size=None):
        self.path = path
        self.train_path = path + '/train.csv'
        self.test_path = path + '/test.csv'
        # self.vocab_path = path + '/imdb.vocab'
        # self.max_vocab_size = max_vocab_size
        # self._read_vocab()
        self.train_text, self.train_y, self.test_text, self.test_y = self.split_yahoo_files()
        print('tokenizing...')

        # Tokenized text of training data
        self.tokenizer = Tokenizer()

        # nlp = spacy.load('en')
        # train_text = [nltk.word_tokenize(doc) for doc in train_text]
        # test_text = [nltk.word_tokenize(doc) for doc in test_text]
        # train_text = [[w.string.strip() for w in nlp(doc)] for doc in train_text]
        # test_text = [[w.string.strip() for w in nlp(doc)] for doc in test_text]
        self.tokenizer.fit_on_texts(self.train_text)
        if max_vocab_size is None:
            max_vocab_size = len(self.tokenizer.word_index) + 1
        # sorted_words = sorted([x for x in self.tokenizer.word_counts])
        # self.top_words = sorted_words[:max_vocab_size-1]
        # self.other_words = sorted_words[max_vocab_size-1:]
        self.dict = dict()
        self.train_seqs = self.tokenizer.texts_to_sequences(self.train_text)
        self.train_seqs2 = [[w if w < max_vocab_size else max_vocab_size for w in doc] for doc in self.train_seqs]

        self.test_seqs = self.tokenizer.texts_to_sequences(self.test_text)
        self.test_seqs2 = [[w if w < max_vocab_size else max_vocab_size for w in doc] for doc in self.test_seqs]

        self.dict['UNK'] = max_vocab_size
        self.inv_dict = dict()
        self.inv_dict[max_vocab_size] = 'UNK'
        self.full_dict = dict()
        self.inv_full_dict = dict()
        for word, idx in self.tokenizer.word_index.items():
            if idx < max_vocab_size:
                self.inv_dict[idx] = word
                self.dict[word] = idx
            self.full_dict[word] = idx
            self.inv_full_dict[idx] = word
        print('Dataset built !')

    def read_yahoo_files(self, filetype):
        texts = []
        labels_index = []  # The index of label of all input sentences, which takes the values 1,2,3,4
        doc_count = 0  # number of input sentences
        path = r'./yahoo_answers_csv/{}.csv'.format(filetype)
        csvfile = open(path, 'r')
        for line in csv.reader(csvfile, delimiter=',', quotechar='"'):
            content = line[1] + ". " + line[2]
            texts.append(content)
            labels_index.append(line[0])
            doc_count += 1

        # Start document processing
        labels = []
        for i in range(doc_count):
            label_class = np.zeros(10, dtype='float32')
            label_class[int(labels_index[i]) - 1] = 1
            labels.append(label_class)

        return texts, labels, labels_index

    def split_yahoo_files(self):
        print("Processing Yahoo Answers! dataset")
        train_texts, train_labels, _ = self.read_yahoo_files('train')  # 120000
        train_texts_small = train_texts[:140000]
        train_labels_small = train_labels[:140000]
        test_texts, test_labels, _ = self.read_yahoo_files('test')  # 7600
        return train_texts_small, train_labels_small, test_texts, test_labels

    def save(self, path='imdb'):
        with open(path + '_train_set.pickle', 'wb') as f:
            pickle.dump((self.train_text, self.train_seqs, self.train_y), f)

        with open(path + '_test_set.pickle', 'wb') as f:
            pickle.dump((self.test_text, self.test_seqs, self.test_y), f)

        with open(path + '_dictionary.pickle', 'wb') as f:
            pickle.dump((self.dict, self.inv_dict), f)

    def read_text(self, path):
        """ Returns a list of text documents and a list of their labels
        (pos = +1, neg = 0) """
        pos_list = []
        neg_list = []

        pos_path = path + '/pos'
        neg_path = path + '/neg'
        pos_files = [pos_path + '/' + x for x in os.listdir(pos_path) if x.endswith('.txt')]
        neg_files = [neg_path + '/' + x for x in os.listdir(neg_path) if x.endswith('.txt')]

        # pos_list = [open(x, 'r').read().lower() for x in pos_files]
        # neg_list = [open(x, 'r').read().lower() for x in neg_files]
        for file_name in pos_files:
            with open(file_name, 'r', encoding='utf-8') as f:
                pos_list.append(rm_tags(" ".join(f.readlines())))
        for file_name in neg_files:
            with open(file_name, 'r', encoding='utf-8') as f:
                neg_list.append(rm_tags(" ".join(f.readlines())))
        data_list = pos_list + neg_list
        labels_list = [1] * len(pos_list) + [0] * len(neg_list)
        return data_list, labels_list

    def build_text(self, text_seq):
        text_words = [self.inv_full_dict[x] for x in text_seq]
        return ' '.join(text_words)
