import numpy as np
import nltk
import pandas as pd
import sys
nltk.download("punkt")


np.set_printoptions(threshold=sys.maxsize, precision=4)
f = open("Corpus/train.txt", "r", encoding="utf8")
line = f.readline()
vocabulary = set()
num_docs = 0
num_classes = 0
whole_titles = []
whole_docs = []
dict_classes = dict()

##########################################################################
#   This part is for reading all the text file and saving the classes
#       and the text for each document in individual arrays
##########################################################################
while line:
    num_docs += 1
    title, _, text = line.partition("@@@@@@@@@@")
    whole_titles.append(title)
    whole_docs.append(text)
    line = f.readline()

whole_titles = np.asarray(whole_titles)
num_classes = np.unique(whole_titles).__len__()
#   prior probability of each class
n_i = np.zeros(num_classes)
for i, title in enumerate(np.unique(whole_titles)):
    # print("The number of the docs for class %s: %d"%(title, np.sum(whole_titles == title)))
    n_i[i] = np.sum(whole_titles == title)
    dict_classes[title] = i

##########################################################################
#   tokenizing the texts for each document and building a vocabulary
#       of the word types
##########################################################################
#   This is a list of numpy arrays which contains tokens of each document
doc_token = []
for i, doc in enumerate(whole_docs):
    tokens = nltk.word_tokenize(doc)
    temp_set = set()
    for word in tokens:
        vocabulary.add(word)
        temp_set.add(word)
    doc_token.append(temp_set)

##########################################################################
#   in this part we convert the set of vocabs to
##########################################################################
temp_list = []
vocabs_index = dict()
for i, vocab in enumerate(vocabulary):
    temp_list.append(vocab)
    vocabs_index[vocab] = i
vocabulary = np.asarray(temp_list)

##########################################################################
#   We will try to calculate the required counts of the words
##########################################################################
num_vocab = vocabulary.__len__()
n_i_w = np.zeros((num_vocab, num_classes))
n_w = np.zeros(num_vocab)

for i in range(num_docs):
    # print("processing document %d out of %d"%(i+1, whole_docs.__len__()))
    for token in doc_token[i]:
        index = vocabs_index[token]
        n_w[index] += 1
        n_i_w[index, dict_classes[whole_titles[i]]] += 1

print(np.where(n_w == 0))

print(n_i)
print(np.sum(n_i_w, axis=0))

n_w_not = num_docs - n_w
n_i_w_not = n_i - n_i_w

##########################################################################
#   Finally we reached a point to calculate the information gains
##########################################################################
information_gain = np.zeros(num_vocab)
for i, vocab in enumerate(vocabulary):
    val = np.log2(n_i / num_docs)
    val[np.isinf(val)] = 0
    val1 = np.log2(n_i_w[i] / n_w[i])
    val1[np.isinf(val1)] = 0
    val2 = np.log2(n_i_w_not[i] / n_w_not[i])
    val2[np.isinf(val2)] = 0

    information_gain[i] = -np.sum(np.multiply((n_i / num_docs), val)) + \
                          np.multiply((n_w[i] / num_docs), np.sum(np.multiply((n_i_w[i] / n_w[i]), val1))) + \
                          np.multiply((n_w_not[i] / num_docs), np.sum(np.multiply((n_i_w_not[i] / n_w_not[i]), val2)))

selected_indexes = np.argsort(-information_gain)[0:200]
file = open("text_file.txt", "a", encoding="utf8")
for i in selected_indexes:
    file.write(str(vocabulary[i]) + "\t" + str(information_gain[i]) + "\n")
file.close()
show = pd.DataFrame(np.concatenate((vocabulary[selected_indexes].reshape((-1, 1)),
                                   information_gain[selected_indexes].reshape((-1, 1))), axis=1),
                    columns=['words', 'information_gain'])
show = show.round(4)

print(show)
