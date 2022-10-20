# Import Stopwords from NLTK Corpus

```python
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))
print(STOPWORDS)
```

# Function to remove Stopwords

```python
def remove_stopwords(text):
	return " ".join([word for word in str(text).split() if word not in STOPWORDS])
```

# Import PorterStemmer - Stemming the words using PorterStemmer

```python
from nltk.stem import PorterStemmer
import re
ps = PorterStemmer()
```

# Function to stem the words using PorterStemmer

```python
def data_preprocess(text):
    text = text.strip() #removes blank spaces before and after the text
    text = re.sub(r'\n', '', text) #regex to replace the new line characters with empty
    text = text.lower() #lower case conversion
    text = ps.stem(text) #stem the words
    text = remove_stopwords(text)
    return text
```

# Import CountVectorizer - Convert a collection of text documents to a matrix of token counts.

https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

```python
from sklearn.feature_extraction.text import CountVectorizer
corpus = [
'This is the first document.',
'This document is the second document.',
'And this is the third one.',
'Is this the first document?',
]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(X)
```