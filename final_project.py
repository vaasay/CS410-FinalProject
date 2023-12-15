######################
## File Description ##
######################

'''
This file is used to calculate accuracy values. Uses the reader.py file and
bigram_naive_bayes.py file to read the movie review dataset, compute whether the
review is positive or negative, and then the accuracy is computed in this file.
'''

######################
## Import Libraries ##
######################

import sys
import argparse
import configparser
import copy

import reader
import bigram_naive_bayes as nb

#########################
## Accuacy Computation ##
#########################

# helper function to compute accuracy of how many correct and incorrect labels were given in bigram_naive_bayes.py
def compute_accuracies(predicted_labels, dev_labels):
    yhats = predicted_labels
    assert len(yhats) == len(dev_labels), "predicted and gold label lists have different lengths"   # error case
    accuracy = sum([yhats[i] == dev_labels[i] for i in range(len(yhats))]) / len(yhats)
    tp = sum([yhats[i] == dev_labels[i] and yhats[i] == 1 for i in range(len(yhats))])
    tn = sum([yhats[i] == dev_labels[i] and yhats[i] == 0 for i in range(len(yhats))])
    fp = sum([yhats[i] != dev_labels[i] and yhats[i] == 1 for i in range(len(yhats))])
    fn = sum([yhats[i] != dev_labels[i] and yhats[i] == 0 for i in range(len(yhats))])
    return accuracy, fp, fn, tp, tn

# print value and also percentage out of n
def print_value(label, value, numvalues):
   print(f"{label} {value} ({value/numvalues}%)")

# print out performance stats
def print_stats(accuracy, false_positive, false_negative, true_positive, true_negative, numvalues):
    print(f"Accuracy: {accuracy}")
    print_value("False Positive", false_positive,numvalues)
    print_value("False Negative", false_negative,numvalues)
    print_value("True Positive", true_positive,numvalues)
    print_value("True Negative", true_negative,numvalues)
    print(f"total number of samples {numvalues}")

#################
## Main Parser ##
#################

def main(args):
    # loads the data from the naive bayes file
    train_set, train_labels, dev_set, dev_labels = nb.load_data(args.training_dir,args.development_dir,args.stemming,args.lowercase)
    
    # loads the predicted label values
    predicted_labels = nb.bigramBayes(dev_set, train_set, train_labels, 
                                          args.laplace,args.bigram_laplace, args.bigram_lambda,args.pos_prior)

    # computes the accuracies from the compute_accuracies helper function
    accuracy, false_positive, false_negative, true_positive, true_negative = compute_accuracies(predicted_labels,dev_labels)
    nn = len(dev_labels)

    # prints these stats to the terminal
    print_stats(accuracy, false_positive, false_negative, true_positive, true_negative, nn)

# allows testing by taking in arguments from our input - allows tunable parameters
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bigram Naive Bayes')
    parser.add_argument('--training', dest='training_dir', type=str, default = 'data/movie_reviews/train',
                        help='the directory of the training data')
    parser.add_argument('--development', dest='development_dir', type=str, default = 'data/movie_reviews/dev',
                        help='the directory of the development data')

    # When doing final testing, reset the default values below to match your settings in naive_bayes.py
    parser.add_argument('--stemming',dest="stemming", type=bool, default=False,
                        help='Use porter stemmer')
    parser.add_argument('--lowercase',dest="lowercase", type=bool, default=False,
                        help='Convert all word to lower case')
    parser.add_argument('--laplace',dest="laplace", type=float, default = 0.009,
                        help='Laplace smoothing parameter')
    parser.add_argument('--bigram_laplace',dest="bigram_laplace", type=float, default = 0.007,
                        help='Laplace smoothing parameter for bigrams')
    parser.add_argument('--bigram_lambda',dest="bigram_lambda", type=float, default = 0.42,
                        help='Weight on bigrams vs. unigrams')
    parser.add_argument('--pos_prior',dest="pos_prior", type=float, default = 0.8,
                        help='Positive prior, i.e. percentage of test examples that are positive')

    args = parser.parse_args()
    main(args)