import string
import pandas as pd
from collections import Counter

def remove_punctuation(text_original):
    """Removes punctuation from a given text string."""
    text_no_punctuation = text_original.translate(str.maketrans('','', string.punctuation))
    return(text_no_punctuation)

def remove_single_character(text):
    """Removes single-character words from a text string."""
    text_len_more_than1 = ""
    for word in text.split():
        if len(word) > 1:
            text_len_more_than1 += " " + word
    return(text_len_more_than1.strip()) # .strip() to remove leading space

def remove_numeric(text, printTF=False):
    """Removes words containing numeric values."""
    text_no_numeric = ""
    for word in text.split():
        isalpha = word.isalpha()
        if printTF:
            print(" {:10} {:}".format(word, isalpha))
        if isalpha:
            text_no_numeric += " " + word
    return(text_no_numeric.strip()) # .strip() to remove leading space

def text_clean(text_original):
    """Applies a sequence of cleaning operations to a text string."""
    text = remove_punctuation(text_original)
    text = remove_single_character(text)
    text = remove_numeric(text)
    return(text)

def df_word(df_txt):
    """
    Calculates the frequency of each word in the captions DataFrame.
    Prints the vocabulary size and returns a DataFrame with words and their counts.
    """
    vocabulary = []
    for txt in df_txt.caption.values:
        vocabulary.extend(txt.split())
    print('Vocabulary Size: %d' % len(set(vocabulary)))
    ct = Counter(vocabulary)
    dfword = pd.DataFrame({"word":list(ct.keys()),"count":list(ct.values())})
    dfword = dfword.sort_values("count", ascending=False)
    dfword = dfword.reset_index()[["word", "count"]]
    return(dfword)

def add_start_end_seq_token(captions):
    """Adds 'startseq' and 'endseq' tokens to captions."""
    caps = []
    for txt in captions:
        txt = 'startseq ' + txt + ' endseq'
        caps.append(txt)
    return(caps)