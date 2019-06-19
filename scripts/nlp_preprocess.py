from nltk.corpus import stopwords
from gensim.parsing.preprocessing import STOPWORDS
from gensim.parsing import preprocessing as pp
from gensim.parsing.preprocessing import strip_multiple_whitespaces,strip_short
import gensim
from numba import jit, autojit

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

pp_list = [
    lambda x: x.lower(),
    pp.strip_tags,
    pp.strip_multiple_whitespaces,
    pp.strip_punctuation,
    pp.strip_short,
    pp.remove_stopwords,
    pp.strip_numeric
          ]

def tokenizer(line):
    tokens = pp.preprocess_string(line, filters=pp_list)
    return tokens

def remove_custom_stopwords(line,custom_stopwords):
    result = [val for i,val in line if val not in custom_stopwords]
    return result



def replace_slangs(lists):
    result = [slang_hash[word] if word in slang_hash else word for word in lists]
    return result

def get_phrases(lists):
    bigram = gensim.models.Phrases(lists)
    bigram_phraser = gensim.models.phrases.Phraser(bigram)
    tokens_ = bigram_phraser[lists]
    trigram = gensim.models.Phrases(tokens_)
    trigram_phraser = gensim.models.phrases.Phraser(trigram)
    return trigram_phraser,bigram_phraser

       
def lemmatize_process(lists,phrased):
    from wordsegment import load,segment
    load() #Load the Segmentation model
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    from nltk.stem.snowball import SnowballStemmer
    stemmer = SnowballStemmer("english")
    bigram_phraser,trigram_phraser = get_phrases(lists)
    for each in lists:
        inter_res = replace_slangs(each)
        each1 = [segment(word) for word in inter_res]
        each2 = flatten_lists(each1)
        #res = each2
        res= bigram_phraser[each2]
        #res= bigram_phraser[each]
        res = trigram_phraser[res]
        phrased.append([lemmatizer.lemmatize(word, pos='v') for word in res])
    return phrased

def remove_email_new_line(data):
    # Remove Emails
    import re
    data = [re.sub('\S*@\S*\s?', '', str(sent)) for sent in data]
    # Remove new line characters
    data = [re.sub('\s+', ' ', str(sent)) for sent in data]
    # Remove distracting single quotes
    data = [re.sub("\'", "", str(sent)) for sent in data]
    return data

def flatten_lists(lists):
    flat_list = [item for sublist in lists for item in sublist]
    return flat_list
    
def preprocess_phrase(lists):
    phrased=[]
    #train_text_final = clean(lists)
    phrased = lemmatize_process(lists,phrased)
    return phrased
    
def get_train_texts(df,col):
    train_texts = []
    try:
        for index,line in enumerate(df[col]):
            tokens = tokenizer(line)
            tokenz = [val for i, val in enumerate(tokens) if val not in custom_stopwords]
            #tokens = remove_custom_stopwords(line,custom_stopwords)
            train_texts.append(tokenz)
        return train_texts

    except Exception as e:
        print(e)
        
def get_phrased_final(train_texts):
  #train_texts = get_train_texts()
  phrased = preprocess_phrase(train_texts)
  phrased_final =[]
  for line in phrased:
    tmp =[]
    for each in line:
        if each not in my_stop_words:
            tmp.append(strip_short(each))
    phrased_final.append(tmp)
  return phrased_final
    
def multi_table(table_list,HTML):
    ''' Acceps a list of IpyTable objects and returns a table which contains each IpyTable in a cell
    '''
    return HTML(
        '<table><tr style="background-color:white;">' + 
        ''.join(['<td>' + table._repr_html_() + '</td>' for table in table_list]) +
        '</tr></table>'
    )



      
  

