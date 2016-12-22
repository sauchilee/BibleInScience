# Python 3 no virtual environment

from bibles import NIV
from viz import *

import nltk
import numpy
import sklearn
import string
import re
import lda


def tokenize_words(string):
    # tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    tokenizer = nltk.tokenize.regexp.WordPunctTokenizer()
    return tokenizer.tokenize(string)

def tokenize_sentences(string):
    tokenizer = nltk.tokenize.PunktSentenceTokenizer()
    return tokenizer.tokenize(string)

def untokenize(list_of_words):
    return ' '.join(list_of_words)

def remove_digits(string):
    return re.sub(r'\b\d+\b', '', string)

def remove_stopwords(list_of_words):
    stoplist = nltk.corpus.stopwords.words('english') + list(string.punctuation)
    return [word for word in list_of_words if word not in stoplist]

def stem(list_of_words):
    stemmer = nltk.stem.porter.PorterStemmer()
    return [stemmer.stem(word) for word in list_of_words]

def get_most_common_words(list_of_words, cutoff=50):
    return nltk.FreqDist(list_of_words).most_common(cutoff)

def get_longest_words(list_of_words, cutoff=15):
    return list(set([word for word in list_of_words if len(word) > 15]))

def get_ngrams(list_of_words, n):
    return nltk.ngrams(list_of_words, n)

def get_average_sentencelength_per_chapter(list_of_chapters):
    result = []
    for chapter in list_of_chapters:
        result.append(numpy.mean(map(len, tokenize_words(chapter))))
    return result

def get_tfidf_matrix(list_of_words):
    tfidf = (sklearn.
             feature_extraction.
             text.
             TfidfVectorizer(strip_accents='ascii', 
                             ngram_range=(0,3), 
                             stop_words='english'))
    return tfidf.fit_transform(list_of_words)

def get_tdm_matrix(list_of_tokenized_documents):
    cv = (sklearn.
          feature_extraction.
          text.
          CountVectorizer(encoding='uft-16', 
                          stop_words='english'))
    return (cv.get_feature_names(), 
            cv.fit_transform(raw_documents=list_of_tokenized_documents))


def perform_lda(tdm_matrix, n_topics, n_iter):
    model = lda.LDA(n_topics=30, n_iter=2000, random_state=1)
    model.fit(tdm_matrix)
    return model

def print_lda(lda, vocabulary, n_top_words=10):
    topic_word = lda.topic_word_  # model.components_ also works
    for i, topic_dist in enumerate(topic_word):
        topic_words = numpy.array(vocabulary)[numpy.argsort(topic_dist)][:-(n_top_words+1):-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))

if __name__ == '__main__':
    NIV.read_in()
    # plot_bar(get_average_sentencelength_per_chapter(NIV.chap), None, None, None)