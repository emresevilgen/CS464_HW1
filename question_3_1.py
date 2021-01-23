import numpy
import math
import csv
import pandas

vocab = {}
r_vocab = {}
f_vocab = {}
index_of_vocab = 0

tokenized_corpus_file = open('tokenized_corpus.csv', mode='r')
csv_reader = csv.reader(tokenized_corpus_file)

no_of_smss = 0

# create an array of unique tokens with frequency greater than 10
for row in csv_reader:
    no_of_smss += 1
    for token in row:
        if token not in f_vocab.keys():
            f_vocab[token] = 1
        else:
            f_vocab[token] += 1
            if f_vocab[token] > 10 and token not in vocab.keys():
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
        if token in vocab.keys():
            m[index][vocab[token]] += 1
    index += 1

# read the labels
labels = pandas.read_csv('labels.csv', header=None, sep="\n")
labels = numpy.array(labels[0])

def prepare_Tj_pi(train_set, train_labels):
    no_of_features = len(train_set[0])
    t_j_spam = numpy.zeros(no_of_features)
    t_j_ham = numpy.zeros(no_of_features)
    pi_spam = 0
    sms_count = 0
    for index, row in enumerate(train_set):
        label = train_labels[index]
        sms_count += 1
        if label == 0:
            t_j_ham = t_j_ham + row
        else:
            t_j_spam = t_j_spam + row
            pi_spam += 1
    pi_spam = pi_spam / sms_count
    return (t_j_spam, t_j_ham, pi_spam)

def test(test_set, test_labels, t_j_spam, t_j_ham, i, sum_tj_spam, sum_tj_ham, estimated_spam, estimated_ham, total_freq):
    no_of_correct = 0
    no_of_test = 0
    alpha = 1
    sum_tj_spam_new = sum_tj_spam + t_j_spam[i] + alpha
    sum_tj_ham_new = sum_tj_ham + t_j_ham[i] + alpha

    teta_j_spam = (t_j_spam[i] + alpha) / sum_tj_spam_new
    teta_j_ham = (t_j_ham[i] + alpha) / sum_tj_ham_new
    estimated_spam_new = estimated_spam.copy()
    estimated_ham_new = estimated_ham.copy()
    total_freq_new = total_freq.copy()

    log_sum_ham = 0
    log_sum_spam = 0
    if sum_tj_ham != 0:
        log_sum_ham = math.log(sum_tj_ham/sum_tj_ham_new)
    if sum_tj_spam != 0:
        log_sum_spam = math.log(sum_tj_spam/sum_tj_spam_new)

    for sample_index in range(len(test_set)):
        no_of_test += 1

        freq = test_set[sample_index][i]
        if estimated_spam_new[sample_index] != -math.inf:
            if teta_j_spam == 0 and freq != 0:
                estimated_spam_new[sample_index] = -math.inf
            else:
                if freq != 0:
                    estimated_spam_new[sample_index] += freq * math.log(teta_j_spam)
                estimated_spam_new[sample_index] += total_freq_new[sample_index] * log_sum_spam
                
        if estimated_ham_new[sample_index] != -math.inf:
            if teta_j_ham == 0 and freq != 0:
                estimated_ham_new[sample_index] = -math.inf
            else:
                if freq != 0:
                    estimated_ham_new[sample_index] += freq * math.log(teta_j_ham)
                estimated_ham_new[sample_index] += total_freq_new[sample_index] * log_sum_ham

        total_freq_new[sample_index] += freq
        label = test_labels[sample_index]
        if (estimated_spam_new[sample_index] >= estimated_ham_new[sample_index] and label == 1):
            no_of_correct += 1
        elif (estimated_spam_new[sample_index] < estimated_ham_new[sample_index] and label == 0):
            no_of_correct += 1
        
    accuracy = (no_of_correct / no_of_test) * 100
    return (accuracy, sum_tj_spam_new, sum_tj_ham_new, estimated_spam_new, estimated_ham_new, total_freq_new)

# Calculate the T spam and ham values
t_j_spam, t_j_ham, pi_spam = prepare_Tj_pi(m[:4460], labels[:4460])

# Forward selection
feature_list = []
max_accuracy = 0
test_set = m[4460:]
test_labels = labels[4460:]

sum_tj_spam = 0
sum_tj_ham = 0
log_pi_spam = math.log(pi_spam)
log_pi_ham = math.log(1 - pi_spam)
estimated_spam = numpy.full(len(test_set), log_pi_spam)
estimated_ham = numpy.full(len(test_set), log_pi_ham)
total_freq = numpy.zeros(len(test_set))

while len(feature_list) < no_of_features:
    index = -1
    for i in range(no_of_features):
        if i in feature_list:
            continue
        temp_accuracy, temp_sum_tj_spam, temp_sum_tj_ham, temp_estimated_spam, temp_estimated_ham, temp_total_freq = test(test_set, test_labels, t_j_spam, t_j_ham, i, sum_tj_spam, sum_tj_ham, estimated_spam, estimated_ham, total_freq)
        if max_accuracy < temp_accuracy:
            index = i
            max_accuracy = temp_accuracy
            max_sum_tj_ham = temp_sum_tj_ham
            max_sum_tj_spam = temp_sum_tj_spam
            max_estimated_ham = temp_estimated_ham
            max_estimated_spam = temp_estimated_spam
            max_total_freq = temp_total_freq
    if index == -1:
        break
    sum_tj_ham = max_sum_tj_ham
    sum_tj_spam = max_sum_tj_spam
    estimated_spam = max_estimated_spam
    estimated_ham = max_estimated_ham
    total_freq = max_total_freq
    feature_list.append(index)
    # print(str(feature_list) + " - " + str(max_accuracy))

feature_list.sort()

print("Accuracy: " + str(max_accuracy) + "% with features " + str(feature_list))

# Save it as forward_selection.csv
forward_selection_file = open('forward_selection.csv', mode='w', newline="")
for i in feature_list:
    forward_selection_file.write("%i\n" % i)

forward_selection_file.close()
tokenized_corpus_file.close()
