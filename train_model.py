"""
train_model.py
==============
Trains Naive Bayes and Logistic Regression models on the combined
real-world dataset produced by data_collector.py.

Workflow:
  1. Run data_collector.py  -> generates data/spam_combined.csv
  2. Run this script        -> generates models/*.pkl
  3. Run app.py             -> starts the web app
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os
import sys

# ---- NLTK downloads ----
for pkg in ['punkt', 'stopwords', 'punkt_tab']:
    nltk.download(pkg, quiet=True)

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()


def clean_text(text):
    """Full NLP preprocessing pipeline."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', ' url ', text)   # replace URLs
    text = re.sub(r'\$[\d,]+', ' money ', text)          # replace money amounts
    text = re.sub(r'\d+', ' num ', text)                  # replace numbers
    text = re.sub(r'[^a-z\s]', ' ', text)                # remove punctuation
    tokens = nltk.word_tokenize(text)
    tokens = [
        ps.stem(w) for w in tokens
        if w not in stop_words and len(w) > 2
    ]
    return " ".join(tokens)


def load_dataset():
    """Load combined dataset. Falls back to SMS Spam if combined not found."""
    combined_path = 'data/spam_combined.csv'
    sms_path = 'data/spam.csv'

    if os.path.exists(combined_path):
        print(f'Loading combined dataset: {combined_path}')
        df = pd.read_csv(combined_path)
        df.columns = ['label', 'text']
    elif os.path.exists(sms_path):
        print(f'Loading SMS Spam fallback: {sms_path}')
        df = pd.read_csv(sms_path, encoding='latin-1')
        df = df[['v1', 'v2']]
        df.columns = ['label', 'text']
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    else:
        print('ERROR: No dataset found!')
        print('Run: python data_collector.py  (to fetch real-world data)')
        print('  OR place SMS Spam CSV as data/spam.csv')
        sys.exit(1)

    df = df.dropna()
    df['label'] = pd.to_numeric(df['label'], errors='coerce').astype(int)
    df = df[df['label'].isin([0, 1])]
    df = df[df['text'].astype(str).str.len() > 5]
    return df


def main():
    print('\n=== SpamShield Model Trainer ===')

    # 1. Load
    df = load_dataset()
    print(f'Dataset size : {len(df)}')
    print(f'Spam         : {df["label"].sum()}')
    print(f'Ham          : {len(df) - df["label"].sum()}')

    # 2. Preprocess
    print('\nPreprocessing text...')
    df['clean_text'] = df['text'].apply(clean_text)

    X = df['clean_text']
    y = df['label']

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f'Train: {len(X_train)} | Test: {len(X_test)}')

    # 4. TF-IDF
    print('\nBuilding TF-IDF features...')
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),      # unigrams + bigrams
        sublinear_tf=True,       # log scaling
        min_df=2
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # 5. Train models
    print('\nTraining models...')

    nb_clf = MultinomialNB(alpha=0.1)
    nb_clf.fit(X_train_tfidf, y_train)

    lr_clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    lr_clf.fit(X_train_tfidf, y_train)

    # 6. Evaluate
    print('\n=== Model Evaluation ===')
    for name, model in [('Naive Bayes', nb_clf), ('Logistic Regression', lr_clf)]:
        y_pred = model.predict(X_test_tfidf)
        y_prob = model.predict_proba(X_test_tfidf)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        print(f'\n--- {name} ---')
        print(f'Accuracy : {acc:.4f}')
        print(f'ROC-AUC  : {auc:.4f}')
        print(f'Confusion Matrix:')
        print(f'  TN={cm[0][0]}  FP={cm[0][1]}')
        print(f'  FN={cm[1][0]}  TP={cm[1][1]}')
        print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

    # 7. Save
    os.makedirs('models', exist_ok=True)
    joblib.dump(nb_clf, 'models/spam_model_nb.pkl')
    joblib.dump(lr_clf, 'models/spam_model_lr.pkl')
    joblib.dump(tfidf,  'models/tfidf_vectorizer.pkl')
    print('\nModels saved to models/')
    print('Run: python app.py')


if __name__ == '__main__':
    main()
