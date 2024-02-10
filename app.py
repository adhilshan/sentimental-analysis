from flask import Flask, render_template, request
from sentiment_analysis import train_sentiment_model, predict_sentiment

app = Flask(__name__)

# Train the sentiment analysis model
sentiment_model = train_sentiment_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    input_text = request.args.get('input_text', '')
    if input_text:
        # Predict sentiment using the sentiment analysis model
        sentiment_prediction = predict_sentiment(sentiment_model, input_text)
        return render_template('result.html', sentiment=sentiment_prediction)
    else:
        return "Please provide input text in the URL arguments."

if __name__ == '__main__':
    app.run(debug=True)
