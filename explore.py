#
# IMPORTS
#
import re

from collections import Counter

import lda
import textmining
import numpy
import codecs

import matplotlib.pyplot as plt
# get_ipython().magic('matplotlib inline')



def read_and_process_NIV():
    #
    # DATA READIN
    #
    f = codecs.open('data/NIV.txt', encoding='utf-16')

    NIV_base = f.read()

    # cut beginning
    main_part_rx = re.compile(r'Genesis.+?(Genesis.*)', re.DOTALL)


    NIV_tmp = re.findall(main_part_rx, NIV_base)[0]
    main_part_rx = re.compile(r'(Genesis.+?)Genesis.*', re.DOTALL)
    NIV_chapter_index = re.findall(main_part_rx, NIV_base)[0]
    NIV_base = NIV_tmp

    # get chapter index
    NIV_chapter_index = re.sub('\r\n', '\n', NIV_chapter_index).strip().split('\n')

    NIV_chapter_index = numpy.array(NIV_chapter_index)
    # get rid of 'New Testament' headline
    content = NIV_chapter_index[NIV_chapter_index != 'New Testament']

    old = []
    new = []
    for chapter in enumerate(content):
        if chapter[0] < 39:
            old.append(chapter[1])
        else: 
            new.append(chapter[1])


    # insert space after numbers
    NIV_base = re.sub(r'(\d+)([^st|nd|rd|th])', r'\1 \2', NIV_base)
    NIV_base

    # get chapters, distinguished by two new lines
    chapters_rx = re.compile(r'[\r\n]{0,}(.+?)[\r\n]{3}', re.DOTALL)

    NIV_chapters = re.findall(chapters_rx, NIV_base)


    # In[7]:

    import itertools
    NIV_chapters = dict(zip(NIV_chapters[0::2], NIV_chapters[1::2]))


    # In[8]:

    NIV_chapters.keys()

    # In[9]:

    # create new and old testament:
    # assume unique Chapter titles
    NIV_base_old = u''
    NIV_base_new = u''

    for chapter in old:
        NIV_base_old = NIV_base_old + NIV_chapters[chapter]

    for chapter in new:
        NIV_base_new = NIV_base_new + NIV_chapters[chapter]

    NIV_base_old


def get_tokens_word():
    pass


def get_tokens_sentence():
    pass


def stem_text():
    pass


def get_average_sentence_lengths():
    pass


def get_histogram():
    pass




#
# BASIC FREQUENCIES
#
from nltk.tokenize import RegexpTokenizer


tokenizer = RegexpTokenizer(r'\w+')
NIV_words = tokenizer.tokenize(NIV_base)


# In[11]:

len(NIV_words)


# In[12]:

from nltk import FreqDist


fdist1 = FreqDist(NIV_words) 
print (fdist1.most_common(50))


# In[13]:

# get longest words
longest_words = list(set([word for word in NIV_words if len(word) > 15]))

print (longest_words)


# In[14]:

# get ngrams
from nltk import ngrams


n = 5
sixgrams = ngrams(NIV_words, n)

print (Counter(sixgrams).most_common(5))


# In[15]:

#
# SENTENCES AND VERSE LENGTHS
#
from nltk.tokenize import PunktSentenceTokenizer
sentence_tokenizer = PunktSentenceTokenizer()
NIV_sentences = sentence_tokenizer.tokenize(NIV_base)


# In[16]:

len(NIV_sentences)


# In[17]:

sentence_lengths = [len(sentence) for sentence in NIV_sentences]


# In[18]:

plt.plot(sentence_lengths)
plt.show()


# In[19]:

#
# SENTENCES AND VERSE LENGTHS PER TESTAMENT
#
sentences_old = sentence_tokenizer.tokenize(NIV_base_old)
avg_sentence_lengths_old = numpy.median([len(sentence) for sentence in sentences_old])

sentences_new = sentence_tokenizer.tokenize(NIV_base_new)
avg_sentence_lengths_new = numpy.median([len(sentence) for sentence in sentences_new])

print (avg_sentence_lengths_old, avg_sentence_lengths_new)


# In[20]:

#
# SENTENCES AND VERSE LENGTHS PER CHAPTER
#
avg_sentence_lengths_per_chapter = []
for i in range(0, len(content)):
    chapter_tokenized = sentence_tokenizer.tokenize(NIV_chapters[content[i]])
    # avg_sentence_lengths_per_chapter.append(sum([len(sentence) for sentence in chapter_tokenized]) / float(len(chapter_tokenized)))
    avg_sentence_lengths_per_chapter.append(numpy.median([len(sentence) for sentence in chapter_tokenized]))


plt.bar(range(0, len(content)), avg_sentence_lengths_per_chapter)
plt.show()


# In[21]:

#
# TFIDF
#
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(strip_accents='ascii', ngram_range=(0,3), stop_words='english')
NIV_words_tfidf = tfidf.fit_transform(NIV_words)


# In[22]:

NIV_words_tfidf


# In[23]:

from sklearn.cluster import KMeans

content_cluster = KMeans(20)
content_cluster.fit(NIV_words_tfidf)


# In[24]:

result = content_cluster.predict(NIV_words_tfidf)


# In[25]:

plt.plot(result)


# In[26]:

Counter(result)


# In[ ]:




# In[100]:

from nltk.tokenize.regexp import RegexpTokenizer
from nltk.tokenize.regexp import WordPunctTokenizer

regex_tokenizer = RegexpTokenizer(r'\w+')
wp_tokenizer = WordPunctTokenizer()

NIV_base_old_token = regex_tokenizer.tokenize(NIV_base_old)
NIV_chapters_list = list(NIV_chapters.values())
NIV_chapters_list_token = [regex_tokenizer.tokenize(chapter) for chapter in NIV_chapters_list]


# In[101]:

# stopwords
import operator
sorted(Counter(NIV_base_old_token).items(), key=operator.itemgetter(1), reverse=True)




# In[102]:

from nltk.corpus import stopwords

import string
stop = stopwords.words('english') + list(string.punctuation)

from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()


NIV_base_old_token = [stemmer.stem(token) for token in NIV_base_old_token if token not in stop]
NIV_chapters_list_token = [[stemmer.stem(token) for token in chapter if token not in stop] for chapter in NIV_chapters_list_token]


NIV_chapters_list_token


# In[103]:

NIV_base_old_token = ' '.join(NIV_base_old_token)
# todo punctuation?
NIV_chapters_list_token = [' '.join(chapter) for chapter in NIV_chapters_list_token]
# remove digits
NIV_chapters_list_token = [re.sub(r'\b\d+\b', '', chapter) for chapter in NIV_chapters_list_token]
NIV_chapters_list_token


# In[104]:

from sklearn.feature_extraction.text import CountVectorizer


# In[105]:

cv = CountVectorizer(encoding='uft-16', stop_words='english')
#tdm_matrix = cv.fit_transform(raw_documents=NIV_chapters.values())
tdm_matrix = cv.fit_transform(raw_documents=NIV_chapters_list_token)
tdm_matrix.shape


# In[106]:

import lda
model = lda.LDA(n_topics=30, n_iter=2000, random_state=1)
model.fit(tdm_matrix)


# In[107]:

vocab = cv.get_feature_names()
topic_word = model.topic_word_  # model.components_ also works
n_top_words = 10
for i, topic_dist in enumerate(topic_word):
    topic_words = numpy.array(vocab)[numpy.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))


# In[ ]:



