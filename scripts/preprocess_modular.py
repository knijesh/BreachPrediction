#!/usr/bin/env python
# coding: utf-8

# # Load and install necessary packages
# 



import numpy as np
import pandas as pd
import os
import nltk
import matplotlib.pyplot as plt
import re
from nltk.tokenize import word_tokenize
import spacy,warnings
from spacy.tokenizer import Tokenizer
from gensim.parsing.preprocessing import remove_stopwords
warnings.filterwarnings('ignore')


# # Read the Dataset into a dataframe with apt encoding

def read_prepare_data(data_path):
	#Add asset from file system
	complaints_df = pd.read_csv(data_path,encoding='ISO-8859-1')
	#complaints_df.columns
	columns= ['ConcernID','IssueID','ConcernDescription','IssueDescription','ResolutionDescription']
	keywords_df = complaints_df[columns]
	keywords_df['keyword_corpus'] = keywords_df['ConcernDescription']+"."+keywords_df['IssueDescription']+"."+keywords_df['ResolutionDescription']
	keywords_df['keyword_corpus'] = keywords_df.fillna('').sum(axis=1)
	return keywords_df,complaints_df

#remove email
def noemail(string):
    string = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+',' ',string)
    return string

#clean the strings
def cleanstring(string):
    cstring = []
    for x in string:
        if x.isalpha():
            cstring.append(x)
        elif x ==' ':
            cstring.append(x)
        else:
            cstring = cstring
    cstring = ''.join(cstring)
    return cstring

def preprocess(doc):
    doc = nltk.word_tokenize(doc)
    doc = nltk.pos_tag(doc)
    return doc

# remove names
def removeNNP(string):
    doc = []
    for wordy in string:
        if wordy[1]=='NNP':
            doc = doc
        else:
            doc.append(wordy[0])
    return(doc)

def removestop(docs):
    remaining = remove_stopwords(docs)
    return remaining

def tokens(docs):
    tokes = nltk.word_tokenize(docs)
    return tokes
    
#df =df[target]
# source= keywords_df['keyword_corpus']
# target= keywords_df['keyword_tokens']

def clean_preprocess(df,source,target):
	#create new column for tokens
	df[target] = df[source].copy()
	#fill blanks: NOTE: this is not needed with Nijesh's update
	df[target] = df[source].where((pd.notnull(df[target])), "nblank")
	#remove emails
	df[target] = df[target].apply(noemail)
	#clean to only is alpha
	df[target] = df[target].apply(cleanstring)
	nlp = spacy.load("en")
	nlpstop = nlp.Defaults.stop_words
	df[target] = df[target].apply(removestop)
	#tokenize
	df[target] = df[target].apply(tokens)

	return df[target]



