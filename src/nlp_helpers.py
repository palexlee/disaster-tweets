import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOPWORDS = stopwords.words('english')

def load_glove(glove_path):
    """Read GloVe embeddings from input path and return as a dict

    Args:
        glove_path (str): path to file containing glove

    Returns:
        embedding_dict (dict): Dict containing word, embedding pairs
    """
    embedding_dict={}
    with open(glove_path,'r', encoding='utf-8') as f:
        for line in f:
            values=line.split()
            word=values[0]
            vectors=np.asarray(values[1:],'float32')
            embedding_dict[word]=vectors
    f.close()
    return embedding_dict

def generate_ngrams(text, n=1, tokenize=True, remove_stopwords=False):
    """Generate ngrams from input text.

    Args:
        text (str): Input text
        n (int, optional): number of words in each gram. Defaults to 1.
        tokenize (bool): whether to use nltk tokenize to split the text. Defaults to True.
        remove_stopwords (bool): whether to ignore stopwords in ngrams. Defaults to False.

    Returns:
        ngrams (list): List of extracted ngrams
    """
    
    txt_list = word_tokenize(text) if tokenize else text.split()
    txt_list = [word for word in txt_list if word.isalnum()]
    if remove_stopwords:
        txt_list = [word for word in txt_list if word not in STOPWORDS]

    ngrams = ['_'.join(txt_list[i:i+n]).lower() for i in range(len(txt_list)-n+1)]
    return ngrams

def create_corpus(text):
    """Create corpus from text: tokenize the string, remoe stopwords and keep only 
    alphanumerical values

    Args:
        text (str): text to parse

    Returns:
        (list): corpus
    """
    tokens = word_tokenize(text.lower())
    clean_tokens = [token for token in tokens if token not in STOPWORDS and token.isalnum()]
    return clean_tokens