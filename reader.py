######################
## File Description ##
######################

"""
This file is responsible for providing functions for reading the files
"""

######################
## Import Libraries ##
######################

from os import listdir
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm

##################
## Reading Data ##
##################

porter_stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')
bad_words = {'aed','oed','eed'} # these words fail in nltk stemmer algorithm

# uses the tqdm library to output the progress bar to visually see how long it takes to go through the training and development set
def loadDir(name,stemming,lower_case,silently=False):
    # Loads the files in the folder and returns a list of lists of words from
    # the text in each file
    X0 = []
    count = 0
    for f in tqdm(listdir(name),disable=silently):
        fullname = name+f
        text = []
        with open(fullname, 'rb') as f:
            for line in f:
                if lower_case:
                    line = line.decode(errors='ignore').lower()
                    text += tokenizer.tokenize(line)
                    
                else:
                    text += tokenizer.tokenize(line.decode(errors='ignore'))
        if stemming:
            for i in range(len(text)):
                if text[i] in bad_words:
                    continue
                text[i] = porter_stemmer.stem(text[i])
        X0.append(text)
        count = count + 1
    return X0

# loads the movie reviews from the data\movie_reviews folder
def load_dataset(train_dir, dev_dir, stemming=False, lower_case=False, silently=True):

    X0 = loadDir(train_dir + '/pos/',stemming, lower_case, silently)
    X1 = loadDir(train_dir + '/neg/',stemming, lower_case, silently)
    X = X0 + X1
    Y = len(X0) * [1] + len(X1) * [0]

    X_test0 = loadDir(dev_dir + '/pos/',stemming, lower_case,silently)
    X_test1 = loadDir(dev_dir + '/neg/',stemming, lower_case,silently)
    X_test = X_test0 + X_test1
    Y_test = len(X_test0) * [1] + len(X_test1) * [0]

    return X,Y,X_test,Y_test