import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
def to_dense(x):
    return x.toarray()
pipeline = Pipeline([   
    ('transformer',FunctionTransformer(to_dense,accept_sparse=True))
])
data = pd.read_csv('test.csv',index_col=False)
data = data.dropna(subset=['text'])
model = joblib.load('predictor.pkl')
x,y = data['text'], data['sentiment']
pipeline.fit(x)
pred = model.predict(x)
acc = accuracy_score(y,pred)
rec = recall_score(y,pred,average='macro')
prec = precision_score(y,pred,average='macro')
print(f'ACC: {acc}')
print(F"REC: {rec}")
print(f'PREC {prec}')