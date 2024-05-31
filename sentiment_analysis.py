import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

class SentimentAnalyzer:
    def __init__(self):
       
        nltk.download('stopwords')

    
        self.data = pd.read_csv('C:\\Users\\hp\\Desktop\\sentiment\\IMDB Dataset.csv')

        # Data Preprocessing
        self._preprocess_data()

        # Check data balance
        print("Sentiment Distribution:", self.data['sentiment'].value_counts())

        # Vectorize the text data
        self.tfidf = TfidfVectorizer(max_features=5000)
        self.X = self.tfidf.fit_transform(self.data['review']).toarray()
        self.y = self.data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)  # Ensure binary labels

        # Train the model
        self._train_model()

    def _preprocess_text(self, text):
        text = re.sub('<[^<]+?>', '', text)  # Remove HTML tags
        text = re.sub('[^a-zA-Z]', ' ', text)  
        text = text.lower()
        text = text.split()
        ps = PorterStemmer()
        text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
        return ' '.join(text)

    def _preprocess_data(self):
        self.data['review'] = self.data['review'].apply(self._preprocess_text)

    def _train_model(self):
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=101)

       
        self.clf = LogisticRegression(max_iter=200)
        self.clf.fit(X_train, y_train)

       
        y_pred = self.clf.predict(X_test)
        print("Training Accuracy:", self.clf.score(X_train, y_train))
        print("Testing Accuracy:", accuracy_score(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

    def save_model(self, model_filename='sentiment_analysis_model.joblib', vectorizer_filename='tfidf_vectorizer.joblib'):
        # Save the trained model
        joblib.dump(self.clf, model_filename)

        # Save the TF-IDF vectorizer
        joblib.dump(self.tfidf, vectorizer_filename)

    def predict_sentiment(self, new_reviews):
        new_reviews = [self._preprocess_text(review) for review in new_reviews]
        new_X_tfidf = self.tfidf.transform(new_reviews).toarray()
        new_predictions = self.clf.predict(new_X_tfidf)
        return ["positive" if pred == 1 else "negative" for pred in new_predictions]

if __name__ == '__main__':
    # Create an instance of SentimentAnalyzer
    sentiment_analyzer = SentimentAnalyzer()

    # Save the model and TF-IDF vectorizer
    sentiment_analyzer.save_model()

    new_test_data = ["This movie was fantastic!", "I did not like the film at all."]
    new_predictions = sentiment_analyzer.predict_sentiment(new_test_data)
    print("New Test Data Predictions:", new_predictions)
