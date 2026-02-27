import numpy as np
import pandas as pd
import re
import os
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    from transformers import DataCollatorWithPadding
    import torch
    TRANSFORMERS_AVAILABLE = True
except:
    TRANSFORMERS_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except:
    VADER_AVAILABLE = False

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

class SentimentAnalyzer:
    def __init__(self, model_name='distilbert-base-uncased-finetuned-sst-2-english', use_gpu=False):
        self.model_name = model_name
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = 0 if self.use_gpu else -1
        self.pipeline = None
        self.model = None
        self.tokenizer = None
        self.labels = ['NEGATIVE','POSITIVE']
        self.vader = None
        self.is_trained = False
        self._load_model()

    def _load_model(self):
        if TRANSFORMERS_AVAILABLE:
            try:
                self.pipeline = pipeline('sentiment-analysis', model=self.model_name, device=self.device)
                self.model = self.pipeline.model
                self.tokenizer = self.pipeline.tokenizer
                if hasattr(self.model.config, 'id2label'):
                    self.labels = [self.model.config.id2label[i] for i in range(len(self.model.config.id2label))]
                self.is_trained = True
                print(f"Loaded transformer: {self.model_name}")
                return
            except Exception as e:
                print(f"Transformer error: {e}, falling back to VADER.")
        if VADER_AVAILABLE:
            self.vader = SentimentIntensityAnalyzer()
            self.is_trained = True
            print("Loaded VADER.")
        else:
            print("No sentiment backend.")

    def analyze(self, text):
        if not self.is_trained:
            return "UNKNOWN", 0.0
        if self.pipeline:
            res = self.pipeline(text)[0]
            return res['label'], res['score']
        elif self.vader:
            scores = self.vader.polarity_scores(text)
            compound = scores['compound']
            if compound >= 0.05:
                return 'POSITIVE', compound
            elif compound <= -0.05:
                return 'NEGATIVE', -compound
            else:
                return 'NEUTRAL', 0.5
        else:
            return "UNKNOWN", 0.0

    def predict_proba(self, text):
        if self.pipeline:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            return torch.softmax(logits, dim=-1).cpu().numpy()[0]
        elif self.vader:
            scores = self.vader.polarity_scores(text)
            compound = scores['compound']
            if compound >= 0.05:
                return np.array([0, 1-compound, compound])
            elif compound <= -0.05:
                return np.array([-compound, 1+compound, 0])
            else:
                return np.array([0,1,0])
        else:
            return np.array([0.33,0.34,0.33])

    def fine_tune(self, csv_path, text_column='text', label_column='sentiment',
                  output_dir='models/sentiment', epochs=3, batch_size=16):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers required.")
        df = pd.read_csv(csv_path)
        texts = df[text_column].tolist()
        labels = df[label_column].tolist()
        unique = sorted(set(labels))
        label2id = {l:i for i,l in enumerate(unique)}
        id2label = {i:l for l,i in label2id.items()}
        num_labels = len(unique)

        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=num_labels, id2label=id2label, label2id=label2id
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
        labels_enc = [label2id[l] for l in labels]

        import torch
        class Dataset(torch.utils.data.Dataset):
            def __init__(self, enc, lab):
                self.enc = enc
                self.lab = lab
            def __getitem__(self, idx):
                item = {k: torch.tensor(v[idx]) for k,v in self.enc.items()}
                item['labels'] = torch.tensor(self.lab[idx])
                return item
            def __len__(self):
                return len(self.lab)

        train_texts, val_texts, train_lab, val_lab = train_test_split(
            texts, labels_enc, test_size=0.1, random_state=42
        )
        train_enc = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
        val_enc = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

        train_dataset = Dataset(train_enc, train_lab)
        val_dataset = Dataset(val_enc, val_lab)

        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            logging_dir=os.path.join(output_dir,'logs'),
        )

        def compute_metrics(eval_pred):
            preds = np.argmax(eval_pred.predictions, axis=1)
            return {'accuracy': accuracy_score(eval_pred.label_ids, preds)}

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            data_collator=DataCollatorWithPadding(tokenizer)
        )
        trainer.train()
        model.save_pretrained(os.path.join(output_dir,'final_model'))
        tokenizer.save_pretrained(os.path.join(output_dir,'final_model'))
        self.model_name = os.path.join(output_dir,'final_model')
        self.pipeline = pipeline('sentiment-analysis', model=self.model_name, tokenizer=self.model_name, device=self.device)
        self.model = self.pipeline.model
        self.tokenizer = self.pipeline.tokenizer
        self.labels = unique
        self.is_trained = True
        print("Fine-tuning complete.")

sentiment_analyzer = SentimentAnalyzer()