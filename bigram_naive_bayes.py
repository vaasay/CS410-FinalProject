##########################################################
## CS 410 Final Project                                 ##
## Project: Bigram naive bayes model                    ##
## Team: Roshan, Vedang, Vaasay, Keyuan, Jimmie         ##
## Language: Python 3.8                                 ##
## Dependancies: nltk, tqdm, reader, collections, numpy ##
##########################################################

## Program Description ##
'''
In this file, we compute whether a movie review is a positive or negative review by using a mixture model
with contributions from a bigram and unigram naive bayes algorithm. The algorithm is written as such:
Each bigram bi is a sequence of 2 consecutive words from either the training or development set. We take
these pairs of words to compute the following probabilities for the bigram/unigram model:
P(Type = Positive | Words) = (P(Type = Positive) / P(Words)) * P(Word Pair | Type = Positive) where we sum up
all of the word pairs
P(Type = Negative | Words) = (P(Type = Negative) / P(Words)) * P(Word Pair | Type = Negative) where we sum up
all of the word pairs
Then we combine the bigram model and the unigram model into a mixture model using a parameter "lambda", which controls
how much emphasis to give to the unigram model and how much to the bigram model. This equation is as follows:
(1 - lambda)*log[P(Y) * sum of all i(P(wi | Y))] + (lambda)*log[P(Y) * sum of all i(P(bi | Y))] where wi is each 
unigram and bi is each bigram.
'''

######################
## Import Libraries ##
######################

import reader
import math
from tqdm import tqdm
import numpy as npy
from collections import Counter

###############
## Debugging ##
###############

'''
print_values and print_values_bigram are utils for printing values.
used for debugging purposes
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

def print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior):
    print(f"Unigram Laplace: {unigram_laplace}")
    print(f"Bigram Laplace: {bigram_laplace}")
    print(f"Bigram Lambda: {bigram_lambda}")
    print(f"Positive prior: {pos_prior}")

##################
## Loading Data ##
##################

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels

###############
## Main Loop ##
###############

"""
Main function for training and predicting with the bigram mixture model.
    You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
"""
def bigramBayes(dev_set, train_set, train_labels, unigram_laplace=0.009, bigram_laplace=0.007, bigram_lambda=0.42, pos_prior=0.8, silently=False):
    yhats = []
    
    # checking for correct input of parameters
    print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior)
    
    # does calculation of P(Y) + P(w0|Y) + P(w1|Y) +...+ P(wn|Y) for unigram and bigram
    uni_pos, uni_neg = unigramPart(dev_set, train_set, train_labels, unigram_laplace, pos_prior)
    bi_pos, bi_neg = bigramPart(dev_set, train_set, train_labels, bigram_laplace, pos_prior)

    # calculates the mixture model which takes the unigram and bigram to determine if a review is positive or negative
    for mixture_model in range(len(dev_set)):
        positives = mixtureModelMath(mixture_model, bigram_lambda, uni_pos, bi_pos)
        negatives = mixtureModelMath(mixture_model, bigram_lambda, uni_neg, bi_neg)

        # appends a 1 if it's positive, or a 0 if it's negative
        if positives >= negatives:
            yhats.append(1)
        else:
            yhats.append(0)

    return yhats

##########################
## Supporting Functions ##
##########################

# does the math for the mixture model
# explanation of this math can be found on our project report
def mixtureModelMath(mixture_model, bigram_lambda, unigram_prob, bigram_prob):
    result = ((1 - bigram_lambda) * unigram_prob[mixture_model]) + ((bigram_lambda) * bigram_prob[mixture_model])
    return result

# does calculation of P(Y) + P(w0|Y) + P(w1|Y) +...+ P(wn|Y) for unigram
def unigramPart(dev_set, train_set, train_labels, unigram_laplace, pos_prior):
    uni_pos = []
    uni_neg = []
    neg_prior = 1 - pos_prior

    # trains the naive bayes algorithm by going through the training set data
    negative_known_prob, negative_unknown_prob = trainingPhase(train_set, train_labels, unigram_laplace, 0)
    positive_known_prob, positive_unknown_prob = trainingPhase(train_set, train_labels, unigram_laplace, 1)
    
    # goes through the development set to calculate whether the reviews are positive or negative
    for individual_set in dev_set:
        positive_words_prob = 0
        negative_words_prob = 0

        positive_words_prob += npy.log(pos_prior)
        negative_words_prob += npy.log(neg_prior)

        # calculates the probability of each review being positive or negative
        # takes the log function to prevent underflow
        for words in individual_set:
            if words in negative_known_prob:
                negative_words_prob += npy.log(negative_known_prob[words])
            else:
                negative_words_prob += npy.log(negative_unknown_prob)
            
            if words in positive_known_prob:
                positive_words_prob += npy.log(positive_known_prob[words])
            else:
                positive_words_prob += npy.log(positive_unknown_prob)

        # appends the probability of a review being positive or negative into a list
        uni_pos.append(positive_words_prob)
        uni_neg.append(negative_words_prob)

    return uni_pos, uni_neg

# does calculation of P(Y) + P(w0|Y) + P(w1|Y) +...+ P(wn|Y) for bigram
def bigramPart(dev_set, train_set, train_labels, bigram_laplace, pos_prior):
    bi_pos = []
    bi_neg = []
    neg_prior = 1 - pos_prior

    # trains the bigram naive bayes algorithm by going through the training set data
    bigram_negative_known_prob, bigram_negative_unknown_prob = bigramTrainingPhase(train_set, train_labels, bigram_laplace, 0)
    bigram_positive_known_prob, bigram_positive_unknown_prob = bigramTrainingPhase(train_set, train_labels, bigram_laplace, 1)

    # goes through the development set to calculate whether the reviews are positive or negative
    for b_individual_set in dev_set:
        b_positive_words_prob = 0
        b_negative_words_prob = 0

        b_positive_words_prob += npy.log(pos_prior)
        b_negative_words_prob += npy.log(neg_prior)

        # iterates through each word in review
        # takes the log function to account for underflow
        for createTuples in range(len(b_individual_set) - 1):

            # creates a tuple by taking the current word and the next word of the review
            # this then creates the bigram
            dev_set_tuple = tuple((b_individual_set[createTuples], b_individual_set[createTuples + 1]))

            # calculates the probability of each bigram being positive or negative and adds them with previous tuples
            # this then calculates the probability for the entire review by the time createTuples reaches the end
            if dev_set_tuple in bigram_negative_known_prob:
                b_negative_words_prob += npy.log(bigram_negative_known_prob[dev_set_tuple])
            else:
                b_negative_words_prob += npy.log(bigram_negative_unknown_prob)

            if dev_set_tuple in bigram_positive_known_prob:
                b_positive_words_prob += npy.log(bigram_positive_known_prob[dev_set_tuple])
            else:
                b_positive_words_prob += npy.log(bigram_positive_unknown_prob)

        # appends the probability of a review being positive or negative into a list
        bi_pos.append(b_positive_words_prob)
        bi_neg.append(b_negative_words_prob)

    return bi_pos, bi_neg

# goes through training phase for the unigram model
def trainingPhase(train_set, train_labels, laplace_smoothing, type):
    review_num = len(train_labels)
    words = {}
    known_prob = {}
    unknown_prob = 0
    total_words = 0
    total_types = 0

    # calculates the number of words for each word in a positive or negative review - stored in list "words"
    # example -> like: 15, hate: 17, sad: 3
    for i in range(review_num):
        if (train_labels[i] == type):
            for train_word in train_set[i]:
                if train_word in words:
                    words[train_word] += 1
                else:
                    words[train_word] = 1

    # calculates the probability of a word while applying laplace smoothing equation
    # doing this for unknown probability, which are all the unseen words, and for the known probability, which are all the seen words
    total_types = len(words)
    for word_counter in words:
        total_words += words[word_counter]

    for word_counter_new in words:
        known_prob[word_counter_new] = (words[word_counter_new] + laplace_smoothing) / (total_words + (laplace_smoothing * (total_types + 1)))

    unknown_prob = laplace_smoothing / (total_words + (laplace_smoothing * (total_types + 1)))

    return known_prob, unknown_prob

# goes through training phase for the bigram model
def bigramTrainingPhase(train_set, train_labels, laplace_smoothing, type):
    bigramWords = {}
    review_num = len(train_labels)
    known_prob = {}
    unknown_prob = 0
    total_words = 0
    total_types = 0

    # calculates the number of tuples for each tuple in a positive or negative review - stored in list "bigramWords"
    # example -> like, movie: 15, hate, seats: 17, sad, price: 3
    for i in range(review_num):
        if train_labels[i] == type:
            for j in range(len(train_set[i]) - 1):
                # takes the current word in a review and the next word in the same review and stores it in a tuple
                eachTuple = tuple((train_set[i][j], train_set[i][j + 1]))

                # checks to see if the tuple has already been seen before or not
                if eachTuple in bigramWords:
                    bigramWords[eachTuple] += 1
                else:
                    bigramWords[eachTuple] = 1

    # calculates the probability of a tuple while applying laplace smoothing equation
    # doing this for unknown probability, which are all the unseen tuples, and for the known probability, which are all the seen tuples
    total_types = len(bigramWords)
    for word_counter in bigramWords:
        total_words += bigramWords[word_counter]

    for word_counter_new in bigramWords:
        known_prob[word_counter_new] = (bigramWords[word_counter_new] + laplace_smoothing) / (total_words + (laplace_smoothing * (total_types + 1)))

    unknown_prob = laplace_smoothing / (total_words + (laplace_smoothing * (total_types + 1)))

    return known_prob, unknown_prob