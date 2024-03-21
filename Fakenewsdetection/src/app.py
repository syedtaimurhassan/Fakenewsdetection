from flask import Flask, jsonify, request
from joblib import load
import numpy as np
import spacy
from textblob import TextBlob
import torch

import spacy
from textblob import TextBlob
from transformers import AutoModel, AutoTokenizer
import numpy as np
from sklearn.preprocessing import StandardScaler

class NamedEntityRecognizer:
    def __init__(self, model='en_core_web_sm'):
        self.nlp = spacy.load(model)

    def recognize(self, text):
        doc = self.nlp(text)
        return doc

class POSTagger:
    def __init__(self, model='en_core_web_sm'):
        self.nlp = spacy.load(model)

    def tag(self, text):
        doc = self.nlp(text)
        return doc

class SentimentAnalyzer:
    def analyze(self, text):
        sentiment = TextBlob(text).sentiment
        return sentiment.polarity, sentiment.subjectivity

class BertEmbedder:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=50)
        outputs = self.model(**inputs)
        sentence_embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        return sentence_embedding[0]

class FeatureExtractor:
    def __init__(self, ner, pos_tagger, sentiment_analyzer, bert_embedder, scaler):
        self.ner = ner
        self.pos_tagger = pos_tagger
        self.sentiment_analyzer = sentiment_analyzer
        self.bert_embedder = bert_embedder
        self.scaler = scaler

    def extract_features(self, text):
        # Named Entity Recognition
        named_entities = self.ner.recognize(text)

        # POS Tagging
        pos_tag = self.pos_tagger.tag(text)

        # Extract POS tags from spacy Token objects
        pos_tags = [token.pos_ for token in pos_tag]
        # Sentiment Analysis
        sentiment = self.sentiment_analyzer.analyze(text)

        # BERT Embedding
        bert_embedding = self.bert_embedder.embed(text)

        # Count POS Tags and Named Entities
        pos_tag_counts = [pos[1] for pos in pos_tags]
        entity_counts = [ent.label_ for ent in named_entities.ents]

        # Combine all features
        features = {
            'pos_tags': pos_tag_counts,
            'named_entities': entity_counts,
            'sentiment': sentiment,
            'bert_embedding': bert_embedding,
        }

        return features


# Create the feature extractor

def predict_statement(statement):
    # Extract features from the statement
    extracted_features = feature_extractor.extract_features(statement)

    # Transform features for the model
    ner_features = [extracted_features['named_entities'].count(tag) for tag in ner_tags]
    pos_features = [extracted_features['pos_tags'].count(tag) for tag in pos_tags]
    sentiment_features = [extracted_features['sentiment'][0], extracted_features['sentiment'][1]]
    bert_features = extracted_features['bert_embedding']

    # Combine NER and POS features
    ner_and_pos_features = np.hstack((ner_features, pos_features))

    # Scale NER and POS features only
    ner_and_pos_features_scaled = scaler.transform([ner_and_pos_features])

    # Ensure sentiment_features and bert_features are 2D
    sentiment_features = np.array(sentiment_features).reshape(1, -1)
    bert_features = np.array(bert_features).reshape(1, -1)

    # Combine all features
    combined_features = np.hstack((ner_and_pos_features_scaled, sentiment_features, bert_features))

    # Predict and get probability
    prediction = rf_classifier.predict(combined_features)
    prediction_probability = rf_classifier.predict_proba(combined_features)

    return prediction[0], prediction_probability[0]



    


# Initialize Flask app
app = Flask(__name__)

    # Initialize your classes
ner = NamedEntityRecognizer()
pos_tagger = POSTagger()
sentiment_analyzer = SentimentAnalyzer()
bert_embedder = BertEmbedder()
scaler = load("")  # Load or initialize your trained scaler

# Define NER and POS tags
ner_tags = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']
pos_tags = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'SPACE']


# Create the feature extractor
feature_extractor = FeatureExtractor(ner, pos_tagger, sentiment_analyzer, bert_embedder, scaler)


# Load or initialize your trained Random Forest model
rf_classifier = load("/weights")  # loading your trained model


@app.route('/predict', methods=['POST'])

def predict():
    data = request.json
    news_text = data['text']

    # Extract features and make a prediction
    prediction, probability = predict_statement(news_text)

    # Check if prediction is scalar or array-like and handle accordingly
    if np.isscalar(prediction):
        prediction_value = int(prediction)  # If it's a scalar, use it directly
    else:
        prediction_value = int(prediction[0])  # If it's array-like, access the first element

    # Format the probability message
    probability_message = ', '.join([f"{prob:.2f}" for prob in probability])

    # Prepare the prediction message
    prediction_message = f"Legitimate News probability {probability_message}" if prediction_value == 1 else f"Fake News with probability {probability_message}"

    # Format the response
    result = {"status": "success", "prediction": prediction_message}
    return jsonify(result)



if __name__ == '__main__':
    app.run(debug=True)
