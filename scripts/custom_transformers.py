import re
import numpy as np
import pandas as pd
import string
import bisect
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from datetime import datetime, timedelta
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, make_union
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from wordsegment import load,segment
#load() #Load the Segmentation model
from nltk.stem import WordNetLemmatizer
from gensim.parsing.preprocessing import STOPWORDS

slang_hash = {'monies':'money','cust':'customer','cus':'customer','custome':'customer','fos':'financial_ombudsman_service',
              'afca':'australian_financial_complaints_authority', 'gtr':'guarantor','cad':'card','kbonus':'bonus','pch':'primary_card_holder',
             'rat':'rate','didnt':"did not",'feb':'February','isn':'is not'}

custom_bi_trigrams = {'break_free_package','financial_ombudsman_service','frequent_fly_points','eligible_bonus_points',
                      'business_select_package','platinum_card','frequent_flyer_black','bonus_point_offer','frequent_flyer_credit',
                      'reward_program_fee','residential_investment_property_loan','senior_personal_banker'}

stop_words = stopwords.words("english")
stop_words = set(stop_words)
custom_stopwords = ['anz','account','customer','card','bank','complaint','january','february','march','april','may','june','july','august',
                   'september','october','november','december','customers']
#custom_stopwords = ['anz','account','customer','bank','complaint']
my_stop_words = STOPWORDS.union(set(custom_stopwords))


# Download the punkt tokenizer for sentence splitting
import nltk.data
nltk.download('punkt')

# Load the punkt tokenizer
tokeniser = nltk.data.load('tokenizers/punkt/english.pickle')

def tokenizer(x):
    return x

#pip install --user gensim
from gensim.models import word2vec
import logging


class EmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
        
        
class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])





class DataFrameFeatureUnion(BaseEstimator, TransformerMixin):
    """Concats all the dataframe produced by a list of transformers.
    Accepts a list of transformers as a parameter.
    Returns the concatted dataframe with all the transformed dataframes."""
    
    def __init__(self, list_of_transformers):
        self.list_of_transformers = list_of_transformers
        
    def transform(self, X, **transformparamn):
        concatted = pd.concat([transformer.transform(X)
                            for transformer in
                            self.fitted_transformers_], axis=1).copy()
        return concatted

    def fit(self, X, y=None, **fitparams):
        self.fitted_transformers_ = []
        for transformer in self.list_of_transformers:
            fitted_trans = clone(transformer).fit(X, y=None, **fitparams)
            self.fitted_transformers_.append(fitted_trans)
        return self

### Datetime transformers

class DatetimeTransfomer(BaseEstimator, TransformerMixin):
    """Converts a pandas column into the specified datetime format
    and returns the dataframe.
    Accepts a column and its format as parameters."""
    
    def __init__(self, column, datetime_format):
        self.column = column
        self.datetime_format = datetime_format
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, **transform_params):
        datetime_col = pd.to_datetime(X[self.column], format=self.datetime_format)
        return pd.DataFrame(datetime_col, index=X.index, columns=[self.column])

class DatetimeDeltaTransfomer(BaseEstimator, TransformerMixin):
    """Calculates the datetime delta(in days) between two columns.
    Accepts a pair a datetime column names as a parameter
    and returns the delta column. Specify the defaut delta 
    if one of the values is null"""
    
    def __init__(self, date_pair, default_diff=20000):
        self.date_pair = date_pair
        self.default_diff = default_diff
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, **transform_params):
        start = X[self.date_pair[0]]
        end = X[self.date_pair[1]]
        start_cond = start.dtype == np.dtype('datetime64[ns]')
        end_cond = end.dtype == np.dtype('datetime64[ns]')
        conditions = [
            (pd.notnull(start) & pd.notnull(end)),
            (pd.isnull(start) & pd.isnull(end))
        ]
        choices = [
            abs((start - end)/timedelta(days=1)),
            0
        ]
        if start_cond and end_cond:
            date_col = np.select(conditions, choices, default=self.default_diff)
            return pd.DataFrame(date_col, index=X.index, columns=['delta_{}_{}'.format(self.date_pair[0], self.date_pair[1])])

### Numeric transformer

class NumericTransformer(BaseEstimator, TransformerMixin):
    """Converts a pandas column to numeric format.
    Accepts a column name as a parameter and returns the transformed array"""
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, **transform_params):
        col_num = pd.to_numeric(X[self.column])
        return pd.DataFrame(col_num, index=X.index, columns=[self.column])

### Text Cleaning
class TextCleaningTransformer(BaseEstimator, TransformerMixin):
    """
    Cleans the text column and returns the cleaned text
    implemented cleaning methods:
    1. remove punctuation through str.maketrans by turning on remove_punctuation
       in fact this is replacing punctuation with space
    2. remove blank elements by turning on remove_blank
    3. remove tokens in a customized list of words by passing the list to remove_words
    4. remove numbers
    """
    
    def __init__(self, column, remove_number = True, start = None, end = None, remove_punctuation = True, remove_blank = True, remove_words = None,
    wordsegment= True,lem = True):
        self.column = column
        self.remove_number = remove_number
        self.remove_punctuation = remove_punctuation
        self.remove_blank = remove_blank
        self.trantab = str.maketrans(string.punctuation, " "*len(string.punctuation))
        self.remove_words = [] if remove_words == None else remove_words
        #self.stem = stem
        self.lem = lem
        self.wordsegment = wordsegment

    def _remove_number(self, row):
        if self.remove_number:
            return re.sub(r'[0-9]',' ',row)
        else:
            return row

    def _remove_blank(self, row):
        if self.remove_blank:
            return re.sub(r' +',' ',row)
        else:
            return row

    def _remove_words(self, row):
        return ' '.join([token for token in row.split() if token not in self.remove_words])
    
    def _remove_slang(self, row):
        return ' '.join([slang_hash[token] if token in slang_hash else token for token in row.split()])
      

    def _lem(self, row):
        if self.lem:
            lemmatizer = WordNetLemmatizer()
            return ' '.join([lemmatizer.lemmatize(token,pos='v') for token in row.split()])
        else:
            return row
    def _wordsegment(self, row):
        load()
        if self.wordsegment:
            return ' '.join([segment(word) for token in row.split() for word in token])
        else:
            return row
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, **transform_params):
        X_column = X[self.column].str.translate(self.trantab) if self.remove_punctuation else X[self.column]
        documents = X_column.astype('str').apply(self._remove_number).apply(self._remove_blank).apply(self._remove_words).apply(self._remove_slang).apply(self._lem)
        return pd.DataFrame(documents, index=X.index, columns=[self.column])



### Text transformers

class TextTokenizerTransfomer(BaseEstimator, TransformerMixin):
    """
    Tokenizes the bag column according to the separator 
    defined, clean the tokens, and returns the tokenized pandas series
    implemented cleaning methods:
    1. remove punctuation through str.maketrans by turning on remove_punctuation
       in fact this is replacing punctuation with space
    2. remove blank elements by turning on remove_blank
    3. remove tokens in a customized list of words by passing the list to remove_words
    4. slice tokens by index in the final cleaned token list, per row,
       by passing start index and/or end index to start and end
       e.g., start = 0 and end = 5 will do tokens[0:5]
    """
    
    def __init__(self, column, separator=';', start = None, end = None):
        self.column = column
        self.separator = separator
        self.start = start
        self.end = end


    def _splitter(self, row):
        tokens = []
        row = row.lower()
        if self.separator in row:
            tokens.extend(row.split(self.separator))
        else:
            tokens.append(row)
        return tokens

    def _remove_blank(self, tokens):
        return [token for token in tokens if token != '']


    def _slice(self, tokens):
        return tokens[self.start:self.end]

    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, **transform_params):
        X_column = X[self.column]
        documents = X_column.astype('str').apply(self._splitter).apply(self._remove_blank).apply(self._slice)
        return pd.DataFrame(documents, index=X.index, columns=[self.column])




class BagVectorizerTransfomer(BaseEstimator, TransformerMixin):
    """Performs a bag-of-items encoding of the bag column
    Accepts column name as a parameter and returns the encoding array"""
    
    def __init__(self, column, remove_lowfreq = True, threshold_lowfreq = 5):
        self.column = column
        self.vocab = None
        self.uniq_tokens_remained = []
        self.remove_lowfreq = remove_lowfreq
        self.threshold_lowfreq = threshold_lowfreq

    def fit(self, X, y=None, **fit_params):
        tokens = np.concatenate(X[self.column].values)
        uniq_tokens = list(set(tokens))
        uniq_tokens.sort() # to ensure the order is always the same 

        if self.remove_lowfreq:
            vocab_freq = Counter(tokens)
            vocab_remove = [k for k,v in vocab_freq.items() if v < self.threshold_lowfreq] + ['']
        else:
            vocab_remove = []

        uniq_tokens_remained = [token for token in uniq_tokens if token not in vocab_remove]
        vocab = {k: v for v, k in enumerate(uniq_tokens_remained)}
        self.vocab = vocab
        self.uniq_tokens_remained = uniq_tokens_remained
        return self

    def get_feature_names(self):
        #return [self.column + '_{}'.format(token) for token, in self.vocab.keys()]
        return [self.column + '_{}'.format(token) for token in self.uniq_tokens_remained] 
    
    def transform(self, X, **transform_params):
        documents = X[self.column]
        arr = np.zeros((X.index.size, len(self.vocab)))
        for index, doc in enumerate(documents):
            tokens_index = [self.vocab[token] for token in doc if token in self.vocab]
            for tok_idx in tokens_index:
                arr[index, tok_idx] += 1
        return pd.DataFrame(arr, index=X.index, columns=self.get_feature_names())
    

### Word2Vec Transformers

#### Preprocessing

def sentence_to_words(sentence, remove_stopwords=True):
    """Function to convert a sentence to a sequence of words,
    optionally removing stop words.  Returns a list of words."""
    
    # Remove non-letters
    sentence = re.sub("[^a-zA-Z]"," ", sentence)
    
    # Convert words to lower case and split them
    words = sentence.lower().split()

    # Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = my_stop_words
        words = [w for w in words if not w in stops]
    
    return words

def document_to_sentences(document, remove_stopwords=True):
    """ Function to split a document into parsed sentences. 
    Returns a list of sentences, where each sentence is a 
    list of words """
    
    # Use NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokeniser.tokenize(document.strip())

    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call sentence_to_words to get a list of words
            sentences.extend(sentence_to_words(raw_sentence, remove_stopwords))

    return sentences

class TokenizerTransfomer(BaseEstimator, TransformerMixin):
    """Tokenizes a column and returns the tokenized pandas series"""
    
    def __init__(self, column):
        self.column = column
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, **transform_params):
        documents = X[self.column].astype('str').apply(document_to_sentences)
        return pd.DataFrame(documents, index=X.index, columns=[self.column])

class DataFrameTransformer(BaseEstimator,TransformerMixin):
  """Makes Custom modifications to Compliance Dataset"""
  def __init__(self, column):
        self.column = column
    
  def fit(self, X, y=None, **fit_params):
        return self
    
  def transform(self, X, **transform_params):
      text_clean = TextCleaningTransformer(column=self.column,remove_words=my_stop_words)
      tokenise = TokenizerTransfomer(column=self.column)
      df= text_clean.transform(X=X)
      final_d = tokenise.transform(X=df)
      final_d.columns = ['features']
      X['features'] = final_d['features']
      #final_df = X[['Breach','features']]
      X=X.drop(self.column,axis=1)
      #print(pd.DataFrame(X,columns =['features']))
      return pd.DataFrame(X,columns =['features'])
      #return pd.DataFrame(X,columns =['Breach','features'])
      
class ColumnExtractorTransformer(BaseEstimator,TransformerMixin):
    """Makes Custom modifications to Compliance Dataset"""
    def __init__(self, column):
      self.column = column
    
    def fit(self, X, y=None, **fit_params):
      return self
    
    def transform(self, X, **transform_params):
      return X[self.column]
  

class Word2VecTransformer(BaseEstimator, TransformerMixin):
    """Converts a tokenized pandas series into an encoding array
    of the specified size."""

    def __init__(self, column, size=10):
        self.size = size
        self.column = column

    def fit(self, X, y=None, **fit_params):
        model = word2vec.Word2Vec(X[self.column], size=self.size, min_count=0)
        model.init_sims(replace=True)
        self.model = model
        return self
    
    def transform(self, X, **transform_params):
        arr = np.zeros((X.shape[0], self.size))
        for i, doc in enumerate(X[self.column]):
            row = np.zeros(self.size)
            for token in doc:
                if token in self.model.wv:
                    row += self.model.wv[token]
            if len(doc) != 0:
                arr[i] = row/len(doc)
            else:
                arr[i] = row
        col_names = [self.column + '_dim_{}'.format(i) for i in range(self.size)]
        return pd.DataFrame(arr, index=X.index, columns=col_names)


class DropFeatures(BaseEstimator, TransformerMixin):
    """Drops User defined features from the pipeline"""

    def __init__(self, column_list):
        self.column_list = column_list

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        #print(X)
        return X.drop(self.column_list, axis=1)

class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    """ This is a custom label encoder which wraps sklearn's
        LabelEncoder and makes it compatible with its pipeline.
        It takes in a column(list or pandas series) as its input.
        Returns the transformed array
        We keep the null values as is but take care of any 
        unknown classes not present in the training set apprearing 
        in the test set with '<unknown>'
        Note: Beware that no value in your dataset has a string
        called '<unkown>' , if there exists, then in the
        code modify it into anything unique"""
    
    def __init__(self, column):
        self.column = column
        self.encoder = LabelEncoder()
        
    def fit(self, X, y=None, **fit_params):
        original_col_without_nans = X[self.column][pd.notnull(X[self.column])].astype('str')
        self.encoder.fit(original_col_without_nans)
        le_classes = self.encoder.classes_.tolist()
        bisect.insort_left(le_classes, '<unknown>')
        self.encoder.classes_ = le_classes
        return self
    # unique value function
    
    def transform(self, X, **transform_params):
        original_col = X[self.column].get_values()
        original_col_without_nans = X[self.column][pd.notnull(X[self.column])]
        label_col = original_col_without_nans.map(lambda s: '<unknown>' if s not in self.encoder.classes_ else s).astype('str')
        transformed_arr = self.encoder.transform(label_col)
        original_col[pd.notnull(original_col)] = transformed_arr
        return pd.DataFrame(original_col, index=X.index, columns=[self.column]).astype('float')