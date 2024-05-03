# Market Mood 
# # Financial News Sentiment Analyzer

Welcome to the Market Mood Financial News Sentiment Analyzer! This project performs sentiment analysis on financial news articles and extracts named entities (such as company names) using natural language processing (NLP) techniques.

## Overview

The Financial News Sentiment Analyzer analyzes textual data from financial news articles to predict sentiment (positive, negative, neutral) and identify key entities mentioned in the articles. It leverages machine learning models for sentiment classification and spaCy for named entity recognition (NER).

## Features

- **Sentiment Analysis**: Classifies news articles into positive, negative, or neutral sentiment categories.
- **Named Entity Recognition (NER)**: Extracts named entities (e.g., company names) from news articles.
- **Interactive Web Interface**: Provides a user-friendly web interface for inputting news articles and viewing sentiment predictions.

## Setup and Usage

1. **Installation**:
   - Clone the repository:
     ```bash
     git clone https://github.com/kshula/sentiment_analyzer.git
     cd sentiment
     ```
   - Install the required dependencies:
     ```bash
     pip install -r requirements.txt
     ```

2. **Training the Model**:
   - Prepare your financial news dataset in CSV format (e.g., `data/output_file.csv`).
   - Update the dataset path in the code (`sentiment_analysis.py`).
   - Run the training script to train the sentiment analysis model:
     ```bash
     python sentiment_analysis.py
     ```

3. **Running the Streamlit App**:
   - Launch the Streamlit web app to interactively predict sentiment and view named entities:
     ```bash
     streamlit run app.py
     ```

4. **Input and Prediction**:
   - Access the app in your web browser (by default at `http://localhost:8501`).
   - Input a financial news article into the text area and click the "Predict" button.
   - View the predicted sentiment and identified named entities (companies) in the output.

## Project Structure

- **`data/`**: Directory containing input data (e.g., CSV files of financial news articles).
- **`sentiment_analysis.py`**: Python script for training the sentiment analysis model.
- **`app.py`**: Streamlit web app for interactive sentiment prediction and entity recognition.
- **`README.md`**: Documentation and project overview (you're reading it right now!).

## Future Improvements

- Enhance sentiment analysis model with advanced NLP techniques (e.g., aspect-based sentiment analysis).
- Incorporate real-time news data sources and update predictions dynamically.
- Deploy the app as a web service for broader accessibility.

## Contributors

- [Kampamba Shula](https://github.com/kshula) - Lead Developer

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
