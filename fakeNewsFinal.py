
# Import libraries required for prediction
import re
import joblib
from pathlib import Path

# Get the folder where this Python file is located
BASE_DIR = Path(__file__).resolve().parent

# Define paths for the saved model and vectorizer
MODEL_PATH = BASE_DIR / "models" / "fake_news_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"

# Load the previously trained model and TF-IDF vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


# Clean news text using the same preprocessing as training
def clean_text(text):
    # Convert input into a string
    text = str(text)

    # Replace special characters with spaces
    text = re.sub(r"\W", " ", text)

    # Convert all letters to lowercase
    text = text.lower()

    # Return cleaned text
    return text


# Predict whether a news article is fake or real
def predict_news(news_text):
    # Clean the user's input
    cleaned_text = clean_text(news_text)

    # Convert text into numerical TF-IDF features
    news_vector = vectorizer.transform([cleaned_text])

    # Predict the label: 1 = Fake, 0 = Real
    prediction = model.predict(news_vector)[0]

    # Convert the numerical label into readable output
    if prediction == 1:
        return "Fake News"
    else:
        return "Real News"


# Run the program when this file is executed directly
if __name__ == "__main__":
    # Take a news article from the user
    news_text = input("Enter a news article: ")

    # Display the model's prediction
    result = predict_news(news_text)
    print("Prediction:", result)
