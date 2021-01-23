import csv
import numpy

vocab = {}
r_vocab = {}
index_of_vocab = 0

tokenized_corpus_file = open('tokenized_corpus.csv', mode='r')
csv_reader = csv.reader(tokenized_corpus_file)

no_of_smss = 0

# create an array of all unique tokens
for row in csv_reader:
    no_of_smss += 1
    for token in row:
        if token not in vocab.keys():
            vocab[token] = index_of_vocab
            r_vocab[index_of_vocab] = token
            index_of_vocab += 1

no_of_features = len(vocab.keys())

# fill the frequency values of each token for each document
m = numpy.zeros(shape=(no_of_smss, no_of_features), dtype=numpy.int8)

tokenized_corpus_file.seek(0)
index = 0
for row in csv_reader:
    for token in row:
        m[index][vocab[token]] += 1
    index += 1

# save it as feature set.csv using comma as the separator
feature_set_file = open('feature_set.csv', mode='w', newline="")
csw_writer = csv.writer(feature_set_file)
csw_writer.writerows(m)
feature_set_file.close()
tokenized_corpus_file.close()