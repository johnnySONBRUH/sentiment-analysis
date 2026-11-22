import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import accuracy_score, recall_score, precision_score
def to_dense(x):
    return x.toarray()
pipeline = Pipeline(
    [
        ('vectorizer',TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ('transformer',FunctionTransformer(to_dense, accept_sparse=True)),
        ('classifier',MultinomialNB())
    ]
)
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
data = pd.read_csv('train.csv',index_col=False)
data = data.dropna(subset=['selected_text'])
x = data['selected_text']
y = data['sentiment']
pipeline.fit(x,y)
joblib.dump(pipeline,'predictor.pkl',compress=3)