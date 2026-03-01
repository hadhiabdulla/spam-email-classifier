import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = nltk.word_tokenize(text)
    tokens = [ps.stem(w) for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(tokens)

# Load dataset (SMS Spam Collection) - save as data/spam.csv
# Download from: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
df = pd.read_csv('data/spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Preprocess
df['clean_text'] = df['text'].apply(clean_text)
X = df['clean_text']
y = df['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train models
nb_clf = MultinomialNB()
nb_clf.fit(X_train_tfidf, y_train)

lr_clf = LogisticRegression(max_iter=1000)
lr_clf.fit(X_train_tfidf, y_train)

# Evaluate
for name, model in [('Naive Bayes', nb_clf), ('Logistic Regression', lr_clf)]:
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f'\n{name}')
    print(f'Accuracy: {acc:.4f}')
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# Save models
os.makedirs('models', exist_ok=True)
joblib.dump(nb_clf, 'models/spam_model_nb.pkl')
joblib.dump(lr_clf, 'models/spam_model_lr.pkl')
joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')
print('\nModels saved in models/ directory.')
