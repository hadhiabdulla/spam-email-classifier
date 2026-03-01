from flask import Flask, render_template, request, jsonify
import joblib
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from datetime import datetime

app = Flask(__name__)

# Load models and vectorizer
nb_model = joblib.load('models/spam_model_nb.pkl')
lr_model = joblib.load('models/spam_model_lr.pkl')
tfidf = joblib.load('models/tfidf_vectorizer.pkl')

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# In-memory session history
history = []

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = nltk.word_tokenize(text)
    tokens = [ps.stem(w) for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(tokens)

def predict_email(text, algo='nb'):
    clean = clean_text(text)
    vec = tfidf.transform([clean])
    model = nb_model if algo == 'nb' else lr_model
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0][1]
    label = 'Spam' if pred == 1 else 'Not Spam'
    confidence = round(prob * 100 if pred == 1 else (1 - prob) * 100, 2)
    algo_name = 'Naive Bayes' if algo == 'nb' else 'Logistic Regression'
    return label, confidence, algo_name

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        email_text = request.form.get('email_text', '')
        algo = request.form.get('algo', 'nb')
        label, confidence, algo_name = predict_email(email_text, algo)
        record = {
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'text_preview': (email_text[:80] + '...') if len(email_text) > 80 else email_text,
            'label': label,
            'confidence': confidence,
            'algo': algo_name,
        }
        history.insert(0, record)
        history[:] = history[:20]
        result = {
            'email_text': email_text,
            'label': label,
            'confidence': confidence,
            'algo': algo_name,
        }
    spam_count = sum(1 for h in history if h['label'] == 'Spam')
    ham_count = sum(1 for h in history if h['label'] == 'Not Spam')
    return render_template('index.html', result=result, history=history,
                           spam_count=spam_count, ham_count=ham_count)

@app.route('/history')
def history_page():
    return render_template('history.html', history=history)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(force=True)
    message = data.get('message', '')
    algo = data.get('algo', 'nb')
    label, confidence, algo_name = predict_email(message, algo)
    return jsonify({
        'message': message,
        'prediction': label,
        'confidence': confidence,
        'algorithm': algo_name
    })

if __name__ == '__main__':
    app.run(debug=True)
