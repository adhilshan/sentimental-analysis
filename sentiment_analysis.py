import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
from time import sleep

# Set the path to the dataset file
dataset_path = 'C:/Users/Admin/Downloads/archive/training.1600000.processed.noemoticon.csv'

# Set the path to the preprocessed data file in the same directory as the script
preprocessed_data_path = os.path.join(os.path.dirname(__file__), 'preprocessed_data.joblib')
# Function to train the sentiment analysis model
def train_sentiment_model():
    try:
        with open(preprocessed_data_path, 'rb') as file:
            print("Preprocessed data file found. Loading data...")
            train_data, test_data, train_labels, test_labels = joblib.load(file)

            # Use SGDClassifier
            model = Pipeline([
                ('count_vectorizer', CountVectorizer(stop_words='english', max_features=2**12)),
                ('tfidf_transformer', TfidfTransformer()),
                ('classifier', SGDClassifier(max_iter=1000, tol=1e-3))
            ])

            # Fit the model
            model.fit(train_data, train_labels)

            # Make predictions on the test set
            predictions = model.predict(test_data)

            # Calculate accuracy
            accuracy = accuracy_score(test_labels, predictions)
            print(f"Model Accuracy: {accuracy * 100:.2f}%")

    except FileNotFoundError:
        print("Preprocessed data file not found. Preprocessing data...")
        columns = ['polarity', 'id', 'date', 'query', 'user', 'text']
        sentiment140_df = pd.read_csv(dataset_path, encoding='ISO-8859-1', header=None, names=columns)
        sentiment140_df['polarity'] = sentiment140_df['polarity'].map({0: 'negative', 4: 'positive'})

        # Separate features and labels
        X = sentiment140_df['text']
        y = sentiment140_df['polarity']

        # Split the data into training and testing sets
        train_data, test_data, train_labels, test_labels = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Use SGDClassifier
        model = Pipeline([
            ('count_vectorizer', CountVectorizer(stop_words='english', max_features=2**12)),
            ('tfidf_transformer', TfidfTransformer()),
            ('classifier', SGDClassifier(max_iter=1000, tol=1e-3))
        ])

        # Fit the model
        model.fit(train_data, train_labels)

        # Make predictions on the test set
        predictions = model.predict(test_data)

        # Calculate accuracy
        accuracy = accuracy_score(test_labels, predictions)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")

        # Save the preprocessed data for future use
        with open(preprocessed_data_path, 'wb') as file:
            joblib.dump((train_data, test_data, train_labels, test_labels), file)

    return model

# Function to predict sentiment
def predict_sentiment(model, input_text):
    return model.predict([input_text])[0]
# Main function
def main():
    # Train the sentiment analysis model
    sentiment_model = train_sentiment_model()

    # Real-time interaction
    while True:
        user_input = input("Enter your text (or 'exit' to quit): ")

        if user_input.lower() == 'exit':
            break

        # Predict sentiment using the sentiment analysis model
        sentiment_prediction = predict_sentiment(sentiment_model, user_input)
        print(f"Sentiment Analysis Model Predicted sentiment: {sentiment_prediction.capitalize()}\n")
        sleep(1)  # Simulate real-time interaction delay

if __name__ == "__main__":
    main()
