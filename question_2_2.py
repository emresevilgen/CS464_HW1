import csv 
import numpy
import math

feature_set_file = open('feature_set.csv', mode='r')
feature_set_reader = csv.reader(feature_set_file)

labels_file = open('labels.csv', mode='r')
labels_reader = csv.reader(labels_file)

row1 = next(feature_set_reader)
feature_set_file.seek(0)
no_of_features = len(row1)

teta_j_spam = numpy.zeros(no_of_features)
teta_j_ham = numpy.zeros(no_of_features)
pi_spam = 0
sms_count = 0

for index, row in enumerate(feature_set_reader):
    label = int(next(labels_reader)[0])
    temp = numpy.array(list(map(int, row)))
    sms_count += 1

    if (label == 0):
        teta_j_ham = teta_j_ham + temp
    else:
        teta_j_spam = teta_j_spam + temp
        pi_spam += 1

    if (sms_count == 4460):
        break

sum_j_spam = sum(teta_j_spam)
for j in range(len(teta_j_spam)):
    teta_j_spam[j] = teta_j_spam[j] / sum_j_spam

sum_j_ham = sum(teta_j_ham)
for j in range(len(teta_j_ham)):
    teta_j_ham[j] = teta_j_ham[j] / sum_j_ham

pi_spam = pi_spam / sms_count

no_of_correct = 0
no_of_test = 0
for index, row in enumerate(feature_set_reader):
    label = int(next(labels_reader)[0])
    no_of_test += 1

    estimate_spam = math.log(pi_spam)
    estimate_ham = math.log(1-pi_spam)
    for j, freq in enumerate(row):
        if (int(freq) != 0 or teta_j_spam[j] != 0):
            if (teta_j_spam[j] == 0):
                estimate_spam = -math.inf
                break
            else:
                estimate_spam += int(freq) * math.log(teta_j_spam[j])
   
    for j, freq in enumerate(row):
        if (int(freq) != 0 or teta_j_ham[j] != 0):
            if (teta_j_ham[j] == 0):
                estimate_ham = -math.inf
                break
            else:
                estimate_ham += int(freq) * math.log(teta_j_ham[j])

    if (estimate_spam >= estimate_ham and label == 1):
        no_of_correct += 1
    elif (estimate_spam < estimate_ham and label == 0):
        no_of_correct += 1

accuracy = (no_of_correct / no_of_test) * 100
print("Accuracy: " + str(accuracy) + "%")

test_accuracy_file = open('test_accuracy.csv', mode='w', newline="")
test_accuracy_file.write(str(accuracy))
test_accuracy_file.close()
labels_file.close()
feature_set_file.close()





