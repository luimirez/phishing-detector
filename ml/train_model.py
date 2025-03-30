import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os

#Sample dataset: replace or expandwith real phishing data ensuring that have the same entries associated
texts = [
    'Verify your account now to avoid suspension',
    'Click here to reset your password immediately',
    'Congratulations! You’ve won a gift card',
    'Update your billing information now',
    'Meeting agenda attached. See you at 2 PM',
    'Here is the report you asked for',
    'Can you check this proposal before 5pm?',
    'Your Amazon receipt is attached'
]

labels = [1, 1, 1, 1, 0, 0, 0, 0] #Indicating the  1 is phishing and 0 is legitimate

#creating a dataframe
data = pd.DataFrame({'text': texts, 'label': labels})

#Vectorize the text data from the information gathered
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['label']

#We are training the logistic regression model
model = LogisticRegression()
model.fit(X, y)

#We have to save the model and the vectorizer
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump((vectorizer, model), f)
    
print(f"[✔] Model trained and saved to {model_path}")