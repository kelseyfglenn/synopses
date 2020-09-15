import re 
import string

def clean_text(text):
    """
    quick text cleaner for MAL synopses, removes attribution and formats
    used to clean training data for regression. for cleaning generated samples, use clean_generator.py.
    """
    text = re.sub(' \[Written.*?\]', '', text) # remove [Written by ...] 
    text = re.sub(' \(Source.*?\)', '', text) # remove (Source: ...)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text) # remove punctuation
    return text.lower()
