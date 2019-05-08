from pyvi import ViTokenizer
import gensim
from more_itertools import unique_justseen
import yaml
import itertools
import string
import re
import unicodedata

f = open('/opt/projects/MISA.Chatbot/zookeeper/zookeeper/acronym.yaml', 'rt')
data = yaml.safe_load(f)
acronym_words = list(itertools.chain.from_iterable(list(data.values())))


def preprocessing(path):
    lines = open(path, encoding='utf-8-sig').readlines()
    result = []
    for line in lines:
        if line != '\n':
            line = remove_punctuation_and_mutiple_spaces(line.lower())
            line = remove_duplicate_character(line.strip().split(' '))
            line = ViTokenizer.tokenize(' '.join(line))
            result.append(line)

    return result


def remove_punctuation_and_mutiple_spaces(x):
    table = str.maketrans({key: None for key in string.punctuation})
    x = x.translate(table)
    return re.sub(' +', ' ', x)


def remove_duplicate_character(list_string):
    result = []
    for s in list_string:
        if 'oo' in s:
            result.append(s)
        else:
            s = ''.join(list(unique_justseen(list(s))))

            # replace acronym word by correct word
            if s in acronym_words:
                s = [key for key, value in data.items() if s in value][0]

            if s.isalnum() is False:
                continue

            result.append(s)
    return result


print(preprocessing('/opt/projects/MISA.Chatbot/zookeeper/zookeeper/data/aspect/irrelevant.txt'))
