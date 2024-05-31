from flask import Flask, request, render_template
import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__)


model = joblib.load('sentiment_analysis_model.joblib')
tfidf = joblib.load('tfidf_vectorizer.joblib')


def preprocess_text(text):
    text = re.sub('<[^<]+?>', '', text)  
    text = re.sub('[^a-zA-Z]', ' ', text)  
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
    return ' '.join(text)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    processed_review = preprocess_text(review)
    transformed_review = tfidf.transform([processed_review])
    prediction = model.predict(transformed_review)
    prediction_text = 'positive' if prediction[0] == 1 else 'negative'
    

    training_accuracy = 0.9319902843263324
    testing_accuracy = 0.87
    confusion_matrix = [[1296, 244], [146, 1314]]
    classification_report = {
        '0': {'precision': 0.90, 'recall': 0.84, 'f1-score': 0.87, 'support': 1540},
        '1': {'precision': 0.84, 'recall': 0.90, 'f1-score': 0.87, 'support': 1460}
    }
    
    return render_template('result.html', review=review, prediction=prediction_text,
                           training_accuracy=training_accuracy, testing_accuracy=testing_accuracy,
                           confusion_matrix=confusion_matrix, classification_report=classification_report)

if __name__ == '__main__':
    app.run(debug=True)
