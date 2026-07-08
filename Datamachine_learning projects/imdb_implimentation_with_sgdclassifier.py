import pprint

import numpy as np
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

stop = stopwords.words('english')
def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
    text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) \
    + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv) # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label

# exam = next(stream_docs(path='imbd_preprocessed.csv'))
# print(exam)

def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y

vect = HashingVectorizer(decode_error='ignore',
                        n_features=2**21,
                        preprocessor=None,
                        tokenizer=tokenizer)
clf = SGDClassifier(loss='log_loss', random_state=1)
doc_stream = stream_docs(path='imdb_preprocessed.csv')

import pyprind
pbar = pyprind.ProgBar(45)
real_x_train = []
real_y_train = []
classes = np.array([0, 1])
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    real_x_train+=X_train
    real_y_train+=y_train
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()

x_train = vect.transform(real_x_train)
y_train = real_y_train

X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print(f'Accuracy: {clf.score(X_test, y_test):.3f}')

correct = (clf.predict(x_train)==y_train).sum()
print(f'correct predictions on training: {len(y_train)}')
print(f'wrong predictions on training: {45000 - correct}')
print(f'wrong predictions on testing: {5000 - (clf.predict(X_test)==y_test).sum()}')
