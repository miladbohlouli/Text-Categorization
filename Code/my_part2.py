import numpy as np
import pandas as pd

import numpy as np
import nltk
import sys

np.set_printoptions(threshold=sys.maxsize, precision=2, suppress=True)
print_flag = True

##########################################################################
#   This is a function that gets the input text file, reads the
#       whole file and extracts the documents and the titles
#       whether for training set or test set
##########################################################################
def framing_data(address):
    if print_flag:
        print("*********************************\nframing the data\n*********************************")
    f = open(address, "r", encoding="utf8")
    line = f.readline()
    whole_titles = []
    whole_docs = []
    while line:
        title, _, text = line.partition("@@@@@@@@@@")
        whole_titles.append(title)
        whole_docs.append(text)
        line = f.readline()

    whole_titles = np.asarray(whole_titles)
    return whole_docs, whole_titles


##########################################################################
#   In this application we tend to extract the vocabulary and finally
#       return a ndarray(vocab_size, num_classes) that contains the
#       number of the occurances of each vocab on each category.
#       the extractions parameter specifies whether you want to shorten
#       the features or not, if true only 200 of the best are returned
##########################################################################
def feature_extraction(whole_docs, whole_titles, extraction=True):
    if print_flag:
        print("*********************************\nDoing the feature extraction\n*********************************")
    num_docs = len(whole_titles)
    global num_classes
    num_classes = np.unique(whole_titles).__len__()
    global dict_classes
    dict_classes = dict()
    vocabulary = set()
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
    #   in this part we convert the set of vocabs to a ndarray
    ##########################################################################
    temp_list = []
    global vocabs_index
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
        print("processing document %d out of %d"%(i+1, whole_docs.__len__()))
        for token in doc_token[i]:
            index = vocabs_index[token]
            n_w[index] += 1
            n_i_w[index, dict_classes[whole_titles[i]]] += 1

    if not extraction:
        return n_i/num_docs, n_i_w, vocabulary

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
    return n_i/num_docs, n_i_w[selected_indexes], vocabulary[selected_indexes]

##########################################################################
#   In this function we implement
##########################################################################
def bigram_features(whole_docs, whole_titles):
    if print_flag:
        print("*********************************\nDoing the feature extraction for bigrams\n*********************************")
    global num_classes
    num_classes = np.unique(whole_titles).__len__()
    global dict_classes
    dict_classes = dict()
    num_docs = len(whole_titles)
    bigram_vocabulary = set()
    unigram_vocabulary = set()

    #   prior probability of each class
    n_i = np.zeros(num_classes)
    for i, title in enumerate(np.unique(whole_titles)):
        if print_flag:
            print("The number of the docs for class %s: %d"%(title, np.sum(whole_titles == title)))
        n_i[i] = np.sum(whole_titles == title)
        dict_classes[title] = i
    ##########################################################################
    #   tokenizing the texts for each document and building a vocabulary
    #       of the word types and a vocabulary of the bigrams
    ##########################################################################
    #   This is a list of numpy arrays which contains tokens of each document
    doc_token = []
    for i, doc in enumerate(whole_docs):
        tokens = nltk.word_tokenize(doc)
        for token in tokens:
            unigram_vocabulary.add(token)

        bigrams = nltk.ngrams(tokens, 2)
        temp_set = set()
        for i, bigram in enumerate(bigrams):
            bigram_vocabulary.add(bigram)
            temp_set.add(bigram)
        doc_token.append(list(temp_set))

    num_bigrams = len(bigram_vocabulary)
    ##########################################################################
    #   in this part we convert the set of bigram_vocabs to a ndarray
    #       The main advantage of making a dictionary that contains the
    #       indexes of the bigrams is for enhancing the performance of
    #       the algorithm, so there is no need to find the index.
    ##########################################################################
    global bigrams_index
    bigrams_index = dict()
    for i, bigram in enumerate(bigram_vocabulary):
        bigrams_index[bigram] = i
    bigram_vocabulary = list(bigram_vocabulary)

    global vocabs_index
    vocabs_index = dict()
    for i, vocab in enumerate(unigram_vocabulary):
        vocabs_index[vocab] = i
    unigram_vocabulary = np.asarray(list(unigram_vocabulary))

    ##########################################################################
    #   We will try to calculate the required counts of the words
    ##########################################################################
    n_i_w = np.zeros((num_bigrams, num_classes))

    for i, doc in enumerate(doc_token):
        if print_flag:
            print("Processing doc %d out of %d"%(i+1, num_docs))
        for bigram in doc:
            n_i_w[bigrams_index[bigram], dict_classes[whole_titles[i]]] += 1

    return n_i / num_docs, n_i_w, bigram_vocabulary

##########################################################################
#   In this application we take the trained matrix of vocabs and
#       apply the absolute discounting smoothing
##########################################################################
def absolute_discounting(trained_matrix, discount_factor):
    if print_flag:
        print("*********************************\nApplying the absolute discounting\n*********************************")
    for i in range(num_classes):
        trained_matrix[np.where(trained_matrix[:, i] != 0), i] -= discount_factor
        trained_matrix[trained_matrix[:, i] == 0, i] += (discount_factor * np.sum(trained_matrix[:, i] != 0)) / np.sum(trained_matrix[:, i] == 0)
    return trained_matrix


##########################################################################
#   This is the main part of the classification which is done using
#       Naive_bayes. The inputs are
#           1. trained_matrix: contains the frequency of each vocab on
#               each individual class
#           2. test_docs: contains the test documents generated by the
#               framing_data function
#           3. mode: specifies the modeling mode of the algorithm,
#               1(default) for unigram and 2 for bigram
##########################################################################
def Naive_bayes(priors, trained_matrix, vocabulary, test_docs, mode=1, discounting_factor=0.1):
    if print_flag:
        print("*********************************\nrunning the Naive Bayes algorithm\n*********************************")
    num_test_docs = len(test_docs)
    num_classes = trained_matrix.shape[1]
    temp_predictaion = np.zeros((num_test_docs, num_classes), dtype=float) + np.log2(priors)

    if mode == 1:
        trained_matrix = absolute_discounting(trained_matrix, discounting_factor)
        likelihood = trained_matrix / np.sum(trained_matrix, axis=0)
        for i, doc in enumerate(test_docs):
            print("Processing document %d out of %d" % (i + 1, num_test_docs))
            tokens = nltk.word_tokenize(doc)
            for token in tokens:
                if token in vocabulary:
                    # temp_predictaion[i] = np.add(temp_predictaion[i], likelihood[vocabs_index[token]])
                    temp_predictaion[i] = np.add(temp_predictaion[i], np.log2(likelihood[np.where(vocabulary == token)]))
    if mode == 2:
        num_vocabs = len(vocabs_index)
        n_w = np.zeros((num_vocabs, num_classes))

        #   If you wanna try with extraction change vocabulary to vocabs_index
        for bigram in vocabulary:
            n_w[vocabs_index[bigram[0]]] += trained_matrix[bigrams_index[bigram]]

        n_w = absolute_discounting(n_w, discounting_factor)
        trained_matrix = absolute_discounting(trained_matrix, discounting_factor)


        for i, doc in enumerate(test_docs):
            if print_flag:
                print("Processing document %d out of %d"%(i+1, num_test_docs))
            tokens = nltk.word_tokenize(doc)
            bigrams = nltk.ngrams(tokens, 2)
            for bigram in bigrams:
                if bigram in bigrams_index:
                    temp_predictaion[i] += np.log2(trained_matrix[bigrams_index[bigram]] / n_w[vocabs_index[bigram[0]]])


    predicted_titles = np.argmax(temp_predictaion, axis=1)
    return predicted_titles


##########################################################################
#   Evaluating the results: 1. Confusion matrix
#                           2. Precision
#                           3. Recall
#                           4. F1-measure
##########################################################################
def evaluation(predicted_labels, ground_truth):
    num_classes = np.unique(ground_truth).__len__()
    confusion_matrix = np.zeros((num_classes, num_classes))
    recall = np.zeros(num_classes)
    precision = np.zeros(num_classes)
    f1_measure = np.zeros(num_classes)
    for i in range(num_classes):
        for j in range(num_classes):
            confusion_matrix[i, j] = np.sum(np.logical_and(predicted_labels == i, ground_truth == j))
        recall[i] = confusion_matrix[i, i] / np.sum(confusion_matrix[:, i])
        precision[i] = confusion_matrix[i, i] / np.sum(confusion_matrix[i, :])
        f1_measure[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])

    return confusion_matrix, precision, recall, f1_measure


##########################################################################
#   Main part of the application
##########################################################################
model = 2
discounting_factor = 0.5

train_docs, train_titles = framing_data("Corpus/train.txt")
if model == 1:
    priors, matrix, vocabulary = feature_extraction(train_docs, train_titles, extraction=True)
#   Now matrix contains the 200 vocabs that have the most information gain
if model == 2:
    priors, matrix, vocabulary = bigram_features(train_docs, train_titles)

global test_titles
test_docs, test_titles = framing_data("Corpus/test.txt")
predicted_titles = Naive_bayes(priors, matrix, vocabulary, test_docs, model, discounting_factor)

#   converting the true labels to numbers for convenience
for i, label in enumerate(test_titles):
    test_titles[i] = dict_classes[label]
conf, precision, recall, f1_measure = evaluation(predicted_titles.astype(int), test_titles.astype(int))

print("The results for the Naive Bayes with discount factor:%.1f and model:%d"%(discounting_factor, model))
print("*********Confusion matrix*********")
print(conf)
print("**************Precision**************")
print(precision)
print("average:%f"%(np.average(precision)))
print("**************recall**************")
print(recall)
print("average:%f"%(np.average(recall)))
print("**************f1_measure**************")
print(f1_measure)
print("average:%f"%(np.average(f1_measure)))
