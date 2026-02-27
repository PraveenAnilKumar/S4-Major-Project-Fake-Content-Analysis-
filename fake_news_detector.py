import numpy as np
import pandas as pd
import re
import string
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

# Transformer support
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except:
    TRANSFORMERS_AVAILABLE = False

# Traditional ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class FakeNewsDetector:
    def __init__(self, use_transformer=False, model_name='distilbert-base-uncased'):
        self.use_transformer = use_transformer and TRANSFORMERS_AVAILABLE
        self.model_name = model_name
        self.transformer_model = None
        self.tokenizer = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Fallback traditional models
        self.vectorizer = None
        self.classifier = None
        self.ensemble_models = []
        self.is_trained = False

    def clean_text(self, text):
        """Basic text cleaning for traditional ML."""
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = re.sub(f"[{string.punctuation}]", " ", text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def train(self, csv_path, text_column='text', label_column='label',
              test_size=0.2, save_path='models/fake_news/', epochs=3, batch_size=16):
        """Main training method – selects transformer or traditional."""
        df = pd.read_csv(csv_path)
        texts = df[text_column].tolist()
        labels = df[label_column].tolist()

        if self.use_transformer:
            self._train_transformer(texts, labels, test_size, save_path, epochs, batch_size)
        else:
            self._train_traditional(texts, labels, test_size, save_path)

        self.is_trained = True

    def _train_transformer(self, texts, labels, test_size, save_path, epochs, batch_size):
        """Train a transformer‑based model (DistilBERT, etc.)."""
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
        import torch
        from torch.utils.data import Dataset

        class NewsDataset(Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels
            def __getitem__(self, idx):
                item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item
            def __len__(self):
                return len(self.labels)

        # Convert all texts to string and replace NaN with empty string
        texts = [str(t) if pd.notna(t) else "" for t in texts]

        # Remove completely empty texts
        valid_indices = [i for i, t in enumerate(texts) if t.strip() != ""]
        texts = [texts[i] for i in valid_indices]
        labels = [labels[i] for i in valid_indices]

        # Split
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )

        # Tokenize
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        train_enc = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
        val_enc = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

        train_dataset = NewsDataset(train_enc, train_labels)
        val_dataset = NewsDataset(val_enc, val_labels)

        # Model
        num_labels = len(set(labels))
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=num_labels)
        model.to(self.device)

        training_args = TrainingArguments(
            output_dir=save_path,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            logging_dir=os.path.join(save_path, 'logs'),
        )

        def compute_metrics(eval_pred):
            preds = np.argmax(eval_pred.predictions, axis=1)
            return {'accuracy': accuracy_score(eval_pred.label_ids, preds)}

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
        trainer.train()
        model.save_pretrained(os.path.join(save_path, 'final_model'))
        tokenizer.save_pretrained(os.path.join(save_path, 'final_model'))
        self.transformer_model = model
        self.tokenizer = tokenizer
        print("Transformer model trained and saved.")

    def _train_traditional(self, texts, labels, test_size, save_path):
        """Train traditional ML ensemble (TF‑IDF + Random Forest + Gradient Boosting)."""
        # Clean texts
        cleaned = [self.clean_text(t) for t in texts]
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english',
                                           ngram_range=(1,3), min_df=2, max_df=0.95)
        X = self.vectorizer.fit_transform(cleaned)
        X_train, X_val, y_train, y_val = train_test_split(X, labels, test_size=test_size,
                                                           random_state=42, stratify=labels)
        # Train ensemble
        rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=8, learning_rate=0.1, random_state=42)
        gb.fit(X_train, y_train)
        voting = VotingClassifier([('rf',rf), ('gb',gb)], voting='soft')
        voting.fit(X_train, y_train)
        self.classifier = voting
        self.ensemble_models = [rf, gb]
        # Save
        os.makedirs(save_path, exist_ok=True)
        joblib.dump(self.vectorizer, os.path.join(save_path, 'tfidf_vectorizer.pkl'))
        joblib.dump(self.classifier, os.path.join(save_path, 'classifier.pkl'))
        for i, m in enumerate(self.ensemble_models):
            joblib.dump(m, os.path.join(save_path, f'ensemble_model_{i}.pkl'))
        print("Traditional models trained and saved.")

    def load_models(self, model_path='models/fake_news/'):
        """Load pre‑trained models from disk."""
        # Try transformer first
        transformer_path = os.path.join(model_path, 'final_model')
        if os.path.exists(transformer_path):
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                self.tokenizer = AutoTokenizer.from_pretrained(transformer_path)
                self.transformer_model = AutoModelForSequenceClassification.from_pretrained(transformer_path)
                self.use_transformer = True
                self.is_trained = True
                print("✅ Loaded transformer model.")
                return True
            except Exception as e:
                print(f"⚠️ Transformer loading failed: {e}")
                print("Falling back to traditional ML...")
        # Fallback to traditional
        try:
            self.vectorizer = joblib.load(os.path.join(model_path, 'tfidf_vectorizer.pkl'))
            self.classifier = joblib.load(os.path.join(model_path, 'classifier.pkl'))
            # load ensemble if any
            for f in os.listdir(model_path):
                if f.startswith('ensemble_model_'):
                    self.ensemble_models.append(joblib.load(os.path.join(model_path, f)))
            self.use_transformer = False
            self.is_trained = True
            print("✅ Loaded traditional models.")
            return True
        except Exception as e:
            print(f"❌ Error loading traditional models: {e}")
            return False

    def predict(self, texts, return_proba=False):
        """Predict label(s) for given text(s)."""
        if not self.is_trained:
            raise ValueError("Model not trained.")
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        if self.use_transformer:
            # Ensure texts are strings
            texts = [str(t) if pd.notna(t) else "" for t in texts]
            inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.transformer_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            conf = np.max(probs, axis=1)
        else:
            cleaned = [self.clean_text(t) for t in texts]
            X = self.vectorizer.transform(cleaned)
            preds = self.classifier.predict(X)
            if hasattr(self.classifier, 'predict_proba'):
                probs = self.classifier.predict_proba(X)
                conf = np.max(probs, axis=1)
            else:
                conf = preds.astype(float)
        if single:
            return (int(preds[0]), float(conf[0])) if not return_proba else float(probs[0][1])
        else:
            return (preds, conf) if not return_proba else probs[:,1]

    def predict_proba(self, texts):
        """Return probability of being fake (class 1)."""
        return self.predict(texts, return_proba=True)

# Global instance for the app
fake_news_detector = FakeNewsDetector()