# Sentence Paraphrase Classifier

This web application classifies whether two sentences are paraphrases using spaCy for text processing and logistic regression for classification. It provides a user-friendly interface to input sentence pairs and receive predictions.

## Features

- **Sentence Classification**: Classifies sentence pairs as paraphrases or non-paraphrases.
- **Cosine Similarity**: Utilizes cosine similarity between sentence embeddings for classification.
- **User Interface**: Simple web interface to input and display results.

## Technologies

- **Backend**: Python, Flask
- **Text Processing**: spaCy
- **Machine Learning**: scikit-learn (Logistic Regression)
- **Frontend**: HTML (templates)

## Installation

1. **Clone the Repository**
    ```sh
    git clone https://github.com/yourusername/sentence-paraphrase-classifier.git
    cd sentence-paraphrase-classifier
    ```

2. **Install Dependencies**
    ```sh
    pip install -r requirements.txt
    ```

3. **Download spaCy Model**
    ```sh
    python -m spacy download en_core_web_md
    ```

## Usage

1. **Run the Application**
    ```sh
    python app.py
    ```

2. **Open in Browser**
    - Navigate to `http://127.0.0.1:5000/` in your web browser.

3. **Predict Paraphrases**
    - Enter two sentences in the provided form.
    - Click "Submit" to see if the sentences are classified as paraphrases or non-paraphrases.

## Project Structure

sentence-paraphrase-classifier/
├── app.py
├── requirements.txt
├── templates/
│ ├── index.html
│ └── result.html
└── README.md


## Contributing

Contributions are welcome! Please feel free to fork the repository, make changes, and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Credits

Developed by Nikhil.
