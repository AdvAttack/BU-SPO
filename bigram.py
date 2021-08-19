from nltk.corpus import wordnet as wn
import string
import numpy as np
from nltk import ngrams
from collections import Counter
from read_files import split_imdb_files, split_yahoo_csv_files, split_agnews_files
import spacy

nlp = spacy.load('en_core_web_sm')

dataset = "imdb"

if dataset == 'imdb':
    train_texts, train_labels, test_texts, test_labels = split_imdb_files()
elif dataset == 'agnews':
    train_texts, train_labels, test_texts, test_labels = split_agnews_files()
elif dataset == 'yahoo':
    train_texts, train_labels, test_texts, test_labels = split_yahoo_csv_files()

num_train = len(train_texts)
num_test = len(test_texts)
num_total = num_train + num_test
print("The number of train samples: ", num_train)
print("The number of test samples: ", num_test)

bigrams = []
str = ' '
text = str.join(test_texts[:1000])
text = text.translate(str.maketrans('', '', string.punctuation))

ngram_counts = Counter(ngrams(text.split(), 2))  # 2: bigram, 3: trigram
most_freq_bigram = ngram_counts.most_common()  # This can limit to most common 1000 with '.most_common(1000)'
num_total_bigrams = len(most_freq_bigram)
print("The total number of bigrams: ", num_total_bigrams)
connect = '_'
bigram_list = []
bigram_syn = []
for i in range(num_total_bigrams):
    bigram = list(most_freq_bigram[i][0])
    bigram_connect = connect.join(bigram)
    freq = most_freq_bigram[i][1]
    tuple_i = (bigram_connect, freq)
    bigram_list.append(tuple_i)
    syns = wn.synsets(bigram_connect)
    if len(syns) == 0:
        continue
    else:
        wordnet_synonyms = []
        for syn in syns:
            wordnet_synonyms.extend(syn.lemmas())

        synonyms = []
        for wordnet_synonym in wordnet_synonyms:
            spacy_synonym = wordnet_synonym.name().replace('_', ' ')
            synonyms.append(spacy_synonym)

        # synonyms = filter(partial(_synonym_prefilter_fn, token), synonyms)
        synonyms = list(set(synonyms))  # avoid repetition
        input = bigram_connect.replace('_', ' ')
        synonyms = [item for item in synonyms if input.lower() != item.lower()]

        bigram_syn_tup = (bigram_connect, freq, synonyms)
        bigram_syn.append(bigram_syn_tup)
print("The number of bigrams that have synonyms: ", len(bigram_syn))
# for i in range(len(bigram_syn)):

bigram_syn = [item for item in bigram_syn if item[2] != []]  # remove those bigrams with no synonyms
print('The number of bigrams that have synonyms different from itself: ', len(bigram_syn))
bigram_syn = np.array(bigram_syn, dtype=object)
np.save('bigram/bigram_syn_imdb.npy', bigram_syn)
a = list(np.load('bigram/bigram_syn_imdb.npy'))
a_list = []
for i in range(len(a)):
    a_list.append(list(a[i]))

print("all done")
