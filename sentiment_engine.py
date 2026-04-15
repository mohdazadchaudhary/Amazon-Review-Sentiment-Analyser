"""
sentiment_engine.py  —  IRTM Core Engine
==========================================
Two sentiment approaches:
  1. VADER               — lexicon-based, no training, works instantly
  2. Logistic Regression — trained on IMDB 50K, ~90% accuracy

Run once to train:  python sentiment_engine.py
Then open dashboard: streamlit run app.py
"""

import os, re, nltk, joblib
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)

nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)

MODEL_PATH      = 'model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'
DATASET_PATH    = 'IMDB Dataset.csv'

# ── Preprocessing ──────────────────────────────────────────────────────
lemmatizer = WordNetLemmatizer()
STOP_WORDS  = set(stopwords.words('english'))
STOP_WORDS -= {'not', 'no', 'never', 'neither', 'nor', "n't"}  # keep negations!


def clean_text(text: str) -> str:
    """Strip HTML → letters only → lowercase → remove stopwords → lemmatize."""
    text   = re.sub(r'<[^>]+>', ' ', text)
    text   = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = text.lower().split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)


# ── Training ───────────────────────────────────────────────────────────
def train_and_save():
    if not os.path.exists(DATASET_PATH):
        print("=" * 60)
        print("IMDB Dataset.csv not found!")
        print("Download from Kaggle:")
        print("kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
        print("Place in same folder, then re-run this file.")
        print("=" * 60)
        return False

    print("\n[1/4] Loading dataset...")
    df          = pd.read_csv(DATASET_PATH)
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    print(f"      {len(df):,} reviews | Positive: {df['label'].sum():,} | Negative: {(df['label']==0).sum():,}")

    print("\n[2/4] Cleaning text (~1 min)...")
    df['clean'] = df['review'].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df['clean'], df['label'],
        test_size=0.2, random_state=42, stratify=df['label']
    )

    print("\n[3/4] TF-IDF vectorization...")
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=5,
        sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)
    print(f"      Vocabulary: {len(vectorizer.vocabulary_):,} features")

    print("\n[4/4] Training Logistic Regression...")
    model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Accuracy : {accuracy_score(y_test, preds):.4f}")
    print(f"Precision: {precision_score(y_test, preds):.4f}")
    print(f"Recall   : {recall_score(y_test, preds):.4f}")
    print(f"F1 Score : {f1_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds, target_names=['Negative', 'Positive']))

    joblib.dump(model,      MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Saved {MODEL_PATH} and {VECTORIZER_PATH}")
    print("\nNext step: streamlit run app.py")
    return True


# ── Prediction ─────────────────────────────────────────────────────────
def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)
    return None, None


def predict(text: str, model=None, vectorizer=None) -> dict:
    """Run both VADER and LR on any text. Returns structured dict."""
    # VADER — always runs, needs raw text (not cleaned)
    analyzer     = SentimentIntensityAnalyzer()
    vs           = analyzer.polarity_scores(text)
    compound     = vs['compound']
    vader_label  = 'Positive' if compound >= 0.05 else ('Negative' if compound <= -0.05 else 'Neutral')

    result = {
        'vader': {
            'label'   : vader_label,
            'compound': round(compound, 4),
            'pos_pct' : round(vs['pos'] * 100, 1),
            'neg_pct' : round(vs['neg'] * 100, 1),
            'neu_pct' : round(vs['neu'] * 100, 1),
        },
        'logistic_regression': None,
        'agreement': None,
        'final_label': vader_label
    }

    # Logistic Regression — only if model is loaded
    if model is not None and vectorizer is not None:
        vec      = vectorizer.transform([clean_text(text)])
        pred     = model.predict(vec)[0]
        proba    = model.predict_proba(vec)[0]
        lr_label = 'Positive' if pred == 1 else 'Negative'

        result['logistic_regression'] = {
            'label'   : lr_label,
            'pos_pct' : round(float(proba[1]) * 100, 1),
            'neg_pct' : round(float(proba[0]) * 100, 1),
        }
        result['agreement']   = (vader_label == lr_label)
        result['final_label'] = lr_label  # LR is more accurate, use as final verdict

    return result


def top_features(model, vectorizer, n=10) -> dict:
    """Top words that push toward positive or negative."""
    names   = vectorizer.get_feature_names_out()
    coefs   = model.coef_[0]
    top_pos = [(names[i], round(float(coefs[i]), 3)) for i in np.argsort(coefs)[-n:][::-1]]
    top_neg = [(names[i], round(float(coefs[i]), 3)) for i in np.argsort(coefs)[:n]]
    return {'positive_words': top_pos, 'negative_words': top_neg}


if __name__ == '__main__':
    train_and_save()
