## Author: David Reilly

import os
import glob
import re
import xml.etree.ElementTree as ET
import string
import collections
import math


def words(document):
    pattern_4 = re.compile(r'[0-9]')
    pattern_5 = re.compile(r'\W')
    text = pattern_4.sub(' ', document)
    text = pattern_5.sub(' ', text)
    wordlist = map(string.lower, filter(lambda x: len(x) >= 3, text.split()))
    return wordlist

def filelist(pathspec):
    return filter(lambda x: os.path.getsize(x) > 0, glob.glob(pathspec))

def get_text(fileName):
    tree = ET.parse(fileName)
    root = tree.getroot()
    list = []
    list.append(root.find('title').text + ' ')
    for i in range(len(root.find('text'))):
        list.append(root.find('text')[i].text + ' ')
    text = "".join(list)
    return text


def create_indexes(files):
    df = collections.Counter(); tf_map = {}
    for file in files:
        d = get_text(file)
        words_1 = words(d)
        n = len(words_1)
        tf = collections.Counter(words_1)
        for t in tf:
            tf[t] = tf[t] / float(n)
            df[t] += 1
        tf_map[file] = tf
    return (tf_map, df)

def doc_tfidf(tf, df, N):
    tfidf = {}
    for t in tf:
        df_t = df[t] / float(N)
        idf_t = 1.0 / df_t
        tfidf[t] = tf[t] * math.log(idf_t)
    return tfidf

def create_tfidf_map(files):
    (tf_map, df) = create_indexes(files)
    tfidf_map = {}
    n = len(files)
    for file in files:
        tfidf = doc_tfidf(tf_map[file], df, n)
        tfidf_map[file] = tfidf
    return tfidf_map
