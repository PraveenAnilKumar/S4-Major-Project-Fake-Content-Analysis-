# Test deepfake model
import tensorflow as tf
model = tf.keras.models.load_model('models/deepfake_final.h5')
print("Deepfake model loaded")

# Test fake news traditional model (if exists)
import joblib
vec = joblib.load('models/fake_news/tfidf_vectorizer.pkl')
clf = joblib.load('models/fake_news/classifier.pkl')
print("Fake news traditional model loaded")

# Test transformer model (if exists)
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model_path = 'models/fake_news/final_model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
print("Transformer model loaded")