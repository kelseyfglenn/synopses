import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.linear_model import LinearRegression

from clean_text import clean_text


class score_text():

    def __init__(self, cleaner=clean_text):
        self.cleaner = clean_text

    def __call__(self, text, vectorizer, factorizer, predictor):
        self.text = text
        self.text_length = len(self.text)
        self.vectorizer = vectorizer
        self.factorizer = factorizer
        self.predictor = predictor
        
        self.clean()
        self.vectorize()
        self.factorize()
        self.add_length_column()
        return self.predict()

    def clean(self):
        self.text = self.cleaner(self.text)
        
    def vectorize(self):
        # expects iterable of docs, so we need to pass as a list to transform just one
        self.text = self.vectorizer.transform([self.text])
    
    def factorize(self):
        self.text = self.factorizer.transform(self.text)

    def add_length_column(self):
<<<<<<< HEAD
        self.text = np.append(self.text, self.text_length).reshape(1,-1)
=======
        self.text.append(self.text_length)
>>>>>>> 8073aca0ea6748cf4bd647de5448b0f15c0f1270

    def predict(self):
        return self.predictor.predict(self.text)[0].round(2)

def score(text, vectorizer, factorizer, predictor):
    
    scorer = score_text()
    return scorer(text, vectorizer, factorizer, predictor)
