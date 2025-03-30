import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

#We have to create the path or route to the model as we are going to create it
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../ml/model.pkl')

#Loading  the model and create a vectorizer to the system
try:
    with open(MODEL_PATH, 'rb') as f:
        vectorizer, model = pickle.load(f)
except FileNotFoundError:
    raise Exception("Model file not found. Please run ml/train_model.py to train and save the model")

def is_phishing(text: str) -> bool:
    try:
        features = vectorizer.transform([text])
        prediction = model.predict(features)
        return bool(prediction[0]) # we are trying to convert numpy.bool into native bool
    except Exception as e:
        print(f"Prediction error: {e}")
        raise