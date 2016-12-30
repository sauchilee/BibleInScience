# TODOS
# given a dictionary of names: create a graph OK
# build dictionary on your own OK
# test performance on Bible OK
# make visualisation
# include sentiments of sentences
# adjust size of bubbles

import re
from time import time
import numpy
import pandas
import itertools
from collections import Counter

from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize.regexp import WordPunctTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.tag import StanfordNERTagger
NER = StanfordNERTagger("/Users/Philipp/stanford-ner-2015-12-09/classifiers/english.all.3class.distsim.crf.ser.gz", "/Users/Philipp/stanford-ner-2015-12-09/stanford-ner.jar")

def tokenize_sentence(string_):
    tokenizer = PunktSentenceTokenizer()
    return tokenizer.tokenize(string_)

def tokenize_words(string_):
    # tokenizer = RegexpTokenizer(r'\w+')
    tokenizer = WordPunctTokenizer()
    return tokenizer.tokenize(string_)

def tag_names_if(tokenized_string, list_of_names):
    tags = []
    for word in tokenized_string:
        if word in list_of_names:
            tags.append("PERSON")
        else:
            tags.append(None)
    return tags

def tag_names_ner(string):
    string = tokenize_words(string)
    string = NER.tag(string)
    result = list(zip(*string))
    return (list(result[0]), list(result[1]))

def clean_tags(tuple_of_lists):
    result = standardise_tags(tuple_of_lists[0], tuple_of_lists[1])
    result = merge_longer_names(result)
    return result

def tag_names_regex(string, list_of_names):
    regexp = re.compile('|'.join(list_of_names), flags=re.IGNORECASE)
    result = re.sub(regexp, r'<PERSON>\g<0></PERSON>', string)
    return result

def standardise_tags(tokenized_string, list_of_tags):
    assert(len(tokenized_string) == len(list_of_tags))

    list_of_tags = numpy.array(list_of_tags)
    indices = (list_of_tags == "PERSON").nonzero()[0]
    for index in indices:
        tokenized_string[index] = '<PERSON>' + tokenized_string[index] + '</PERSON>'
    return ' '.join(tokenized_string)

def merge_longer_names(tagged_string):
    regexp = re.compile('</PERSON> <PERSON>')
    return re.sub(regexp, r' ', tagged_string)

def extract_names(tagged_string):
    regexp = re.compile(r'<PERSON>(.+?)</PERSON>', re.DOTALL)
    return re.findall(regexp, tagged_string)

def find_twofold_subsets(set_or_list):
    set_ = set(set_or_list)
    return set(itertools.combinations(set_, 2))

def build_name_list(tagged_string):
    names = extract_names(tagged_string)
    return Counter(names, sort=True)

def clean_names(tagged_string, dict_):
    # format: {clean_name: [alternative_name1, alternativ_name2, ...]}
    for clean_name, alternatives in dict_.items():
        print('<PERSON>' + '|'.join(alternatives) + '</PERSON>')
        regexp = re.compile(r'<PERSON>(' + '|'.join(alternatives) + ')</PERSON>')
        tagged_string = re.sub(regexp, r'<PERSON>' + clean_name + '</PERSON>', tagged_string)
    return tagged_string

def build_network(tagged_string):
    sentences = tokenize_sentence(tagged_string)
    edges = []
    for sentence in sentences:
        names = extract_names(sentence)
        if len(names) < 2:
            continue
        elif len(names) == 2:
            edges.append(names)
        elif len(names) > 2:
            names = list(find_twofold_subsets(names))
            edges.extend(names)
    edges = list(map(tuple, map(sorted, edges)))
    return Counter(edges)

def save_network(counter_of_edges, use_top_n=100, filename=False):
    network_df = pandas.DataFrame.from_dict(dict(counter_of_edges), orient='index')
    network_df.reset_index(inplace=True)
    network_df['source'] = network_df['index'].apply(lambda x: x[0])
    network_df['target'] = network_df['index'].apply(lambda x: x[1])
    network_df.drop('index', axis=1, inplace=True)
    network_df.rename(columns={0: 'value'}, inplace=True)
    network_df.sort_values('value', ascending=False, inplace=True)
    if filename is True:
        network_df[:use_top_n].to_csv(filename + '.csv', index=False)
    return network_df[['source', 'target', 'value']]


if __name__ == '__main__':
    test = "Philipp Dufter and Johanna are on holiday. They meet Josef. Josef and Philipp then drink a beer. Johanna calls Katrin and Eckart. Philipp Dufter and Johanna. Phil goes. Joshua son of Nun eats breakfast."
    #t = time()
    #print (tag_names_if(tokenize_words(test), ['Philipp', 'Johanna']))
    #print(time() - t)
    #t = time()
    #print (tag_names_regex(test, ['Philipp', 'Johanna', 'Dufter']))
    #print(time() - t)
    result = clean_tags(tag_names_ner(test))
    raw = tag_names_ner(test)
    print(raw)
    # print(tokenize_sentence(merge_longer_names(standardise_tags(result[0], result[1]))))
    # print(build_name_list(merge_longer_names(standardise_tags(result[0], result[1]))))
    print(build_network(clean_names(result, {"Philipp": ["Philipp Dufter", "Phil"]})))
    #print(clean_names(result, {"Philipp": ["Philipp Dufter", "Phil"]}))



