import re
from nltk.tokenize import RegexpTokenizer
from nltk import FreqDist

from nltk import ngrams

from collections import Counter



import codecs
f = codecs.open('data/NIV.txt', encoding='utf-16')

NIV = f.read()

# TODO get rid of header


# do some cleaning



# get frequencies
tokenizer = RegexpTokenizer(r'\w+')
NIV = tokenizer.tokenize(NIV)

fdist1 = FreqDist(NIV) 
print fdist1.most_common(50)


# get longest words
longest_words = list(set([word for word in NIV if len(word) > 15]))

print longest_words

# get ngrams
n = 3
sixgrams = ngrams(NIV, n)

print Counter(sixgrams).most_common(5)




