import spacy
import numpy as np
from sklearn.linear_model import LogisticRegression
from flask import Flask, render_template, request

# Load spaCy model
nlp = spacy.load("en_core_web_md")

# Example dataset (replace with your own dataset)
example_sentence_pairs = [
    ("I love coding", "Coding is my passion"),
    ("The cat sat on the mat", "The cat is resting on the rug"),
    ("The sun is shining", "It's a sunny day"),
    ("Machine learning is fascinating", "I'm intrigued by machine learning"),
    ("I love coding", "I hate coding"),
    ("The sun is shining", "The moon is shining"),
    ("I enjoy coding", "Coding is a tedious task"),
    ("The cat chased the mouse.", "The mouse was chased by the cat."),
    ("The quick brown fox jumps over the lazy dog.", "The lazy dog is jumped over by the quick brown fox."),
    ("She sells seashells by the seashore.", "The seashells are sold by her near the seashore."),
    ("I cannot go out tonight.", "Tonight, I am unable to go out."),
    ("He is a good student.", "He excels in academics."),
    ("The concert was canceled due to bad weather.", "Bad weather led to the cancellation of the concert."),
    ("The cake tastes delicious.", "Delicious is how the cake tastes."),
    ("The sun rises in the east.", "In the east, the sun rises."),
    ("She plays the piano beautifully.", "Playing the piano, she is beautiful."),
    ("The book is on the table.", "On the table lies the book."),
    # Additional example sentence pair
    ("The sky is blue.", "Blue is the color of the sky.")
]

# Define labels for the example dataset
labels = [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1]  # 1 for paraphrase, 0 for non-paraphrase

# Preprocess text and extract features
def preprocess_and_extract_features(sentence_pairs):
    features = []
    for pair in sentence_pairs:
        doc1 = nlp(pair[0])
        doc2 = nlp(pair[1])
        # Average word embeddings for each sentence
        avg_vector1 = np.mean([token.vector for token in doc1], axis=0)
        avg_vector2 = np.mean([token.vector for token in doc2], axis=0)
        # Cosine similarity between sentence embeddings
        similarity = np.dot(avg_vector1, avg_vector2) / (np.linalg.norm(avg_vector1) * np.linalg.norm(avg_vector2))
        features.append(similarity)
    return np.array(features).reshape(-1, 1)

# Extract features for example dataset
X = preprocess_and_extract_features(example_sentence_pairs)

# Train logistic regression model
model = LogisticRegression()
model.fit(X, labels)

# Set threshold for cosine similarity
threshold = 0.5  # Adjust as needed

# Initialize Flask application
app = Flask(__name__)

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get input sentences from form
    sentence1 = request.form['sentence1']
    sentence2 = request.form['sentence2']

    # Preprocess user input sentences
    doc1 = nlp(sentence1)
    doc2 = nlp(sentence2)
    avg_vector1 = np.mean([token.vector for token in doc1], axis=0)
    avg_vector2 = np.mean([token.vector for token in doc2], axis=0)
    similarity = np.dot(avg_vector1, avg_vector2) / (np.linalg.norm(avg_vector1) * np.linalg.norm(avg_vector2))

    # Make prediction for user input sentences
    if similarity >= threshold:
        prediction = model.predict([[similarity]])
        if prediction == 1:
            result = "Paraphrase"
        else:
            result = "Non-paraphrase"
    else:
        result = "Non-paraphrase"  # Predict non-paraphrase for dissimilar sentences

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
