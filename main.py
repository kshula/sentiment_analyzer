import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder


class TextTokenizer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def tokenize(self, text):
        tokens = word_tokenize(text.lower())
        filtered_tokens = [token for token in tokens if token.isalpha() and token not in self.stop_words]
        return ' '.join(filtered_tokens)  # Join tokens into a single string

class SentimentAnalyzer:
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
        self.sentiments = self.data['sentiment']
        self.texts = self.data['text']
        self.tokenizer = TextTokenizer()
        self.vectorizer = TfidfVectorizer()
        self.label_encoder = LabelEncoder()

    def preprocess_texts(self):
        preprocessed_texts = [self.tokenizer.tokenize(text) for text in self.texts]
        return preprocessed_texts

    def encode_sentiments(self):
        encoded_sentiments = self.label_encoder.fit_transform(self.sentiments)
        return encoded_sentiments

    def train_test_split(self):
        preprocessed_texts = self.preprocess_texts()
        X = self.vectorizer.fit_transform(preprocessed_texts)
        y = self.encode_sentiments()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_and_evaluate_models(self):
        X_train, X_test, y_train, y_test = self.train_test_split()

        models = {
            'Random Forest': RandomForestClassifier(),
            'Logistic Regression': LogisticRegression(),
            'SVM': SVC(),
            'Multinomial Naive Bayes': MultinomialNB(),
            'XGBoost': XGBClassifier()
        }

        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = accuracy

        return results

# Example usage
data_file = 'data\\output_file.csv'
sentiment_analyzer = SentimentAnalyzer(data_file)
model_results = sentiment_analyzer.train_and_evaluate_models()

# Print model accuracies
for model, accuracy in model_results.items():
    print(f"{model}: Accuracy - {accuracy:.4f}")
