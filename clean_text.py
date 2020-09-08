import re 
import string

def clean_text(text):
    """
    quick text cleaner
    """
    text = re.sub(' \[Written.*?\]', '', text) # remove [Written by ...] 
    text = re.sub(' \(Source.*?\)', '', text) # remove (Source: ...)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text) # remove punctuation
    return text.lower()
