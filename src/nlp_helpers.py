from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOPWORDS = stopwords.words('english')

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