import streamlit as st
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder

# Define TextTokenizer class for text preprocessing
class TextTokenizer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def tokenize(self, text):
        tokens = word_tokenize(text.lower())
        filtered_tokens = [token for token in tokens if token.isalpha() and token not in self.stop_words]
        return ' '.join(filtered_tokens)

# Define SentimentAnalyzer class for sentiment analysis and NER
class SentimentAnalyzer:
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
        self.sentiments = self.data['sentiment']
        self.texts = self.data['text']
        self.tokenizer = TextTokenizer()
        self.vectorizer = TfidfVectorizer()
        self.label_encoder = LabelEncoder()
        self.random_forest_model = RandomForestClassifier()
        self.nlp = spacy.load('en_core_web_sm')

    def preprocess_texts(self):
        preprocessed_texts = [self.tokenizer.tokenize(text) for text in self.texts]
        return preprocessed_texts

    def encode_sentiments(self):
        encoded_sentiments = self.label_encoder.fit_transform(self.sentiments)
        return encoded_sentiments

    def train_model(self):
        preprocessed_texts = self.preprocess_texts()
        X = self.vectorizer.fit_transform(preprocessed_texts)
        y = self.encode_sentiments()
        self.random_forest_model.fit(X, y)

    def predict_sentiment(self, news_article):
        preprocessed_article = self.tokenizer.tokenize(news_article)
        article_vector = self.vectorizer.transform([preprocessed_article])
        predicted_label = self.random_forest_model.predict(article_vector)
        sentiment_label = self.label_encoder.inverse_transform(predicted_label)

        # Perform Named Entity Recognition (NER) using spaCy
        doc = self.nlp(news_article)
        named_entities = [ent.text for ent in doc.ents if ent.label_ == 'ORG']

        return sentiment_label[0], named_entities

# Create Streamlit app
def main():
    st.title("Market Mood Sentiment Analysis")

    # Load data and train model
    data_file = 'data\\output_file.csv'
    sentiment_analyzer = SentimentAnalyzer(data_file)
    sentiment_analyzer.train_model()

    # User input for news article
    user_article = st.text_area("Enter a financial news article:", height=200)

    # Predict sentiment and named entities on button click
    if st.button("Predict", key="predict_button"):
        if user_article:
            predicted_sentiment, named_entities = sentiment_analyzer.predict_sentiment(user_article)
            st.subheader("Predicted Sentiment:")
            st.write(predicted_sentiment)

            if named_entities:
                st.subheader("Named Entities (Companies):")
                for entity in named_entities:
                    st.write(f"- {entity}")
        else:
            st.warning("Please enter a financial news article.")

if __name__ == "__main__":
    main()
