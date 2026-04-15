# Amazon Review Sentiment Analyser
## IRTM Mini Project

---

## STEP 1 — Install Python libraries
Open terminal / command prompt in this folder and run:

```
pip install -r requirements.txt
```

---

## STEP 2 — Download the IMDB dataset (for training)

1. Go to: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
2. Download  IMDB Dataset.csv
3. Place it in THIS folder (same folder as app.py)

---

## STEP 3 — Train the model

```
python sentiment_engine.py
```

This takes about 1-2 minutes.
It will print accuracy, F1 score, and save model.pkl and vectorizer.pkl

---

## STEP 4 — (Optional) Add SerpAPI key for real Amazon reviews

1. Sign up free at: https://serpapi.com
2. Copy your API key
3. Open scraper.py
4. Replace  YOUR_SERPAPI_KEY_HERE  with your actual key

If you skip this step, demo data is used automatically — still works for demo!

---

## STEP 5 — Launch the dashboard

```
streamlit run app.py
or
py -3.11 -m streamlit run app.py
```

Browser opens automatically at  http://localhost:8501

---

## What each file does

| File                  | What it does                                    |
|-----------------------|-------------------------------------------------|
| sentiment_engine.py   | VADER + Logistic Regression — the ML brain      |
| scraper.py            | Fetches Amazon reviews using SerpAPI            |
| app.py                | Streamlit dashboard — what professor sees       |
| requirements.txt      | All Python libraries needed                     |
| model.pkl             | Saved trained model (created after Step 3)      |
| vectorizer.pkl        | Saved TF-IDF vectorizer (created after Step 3)  |

---

## IRTM Concepts in this project (for viva)

| Concept              | Where used                              |
|----------------------|-----------------------------------------|
| Text preprocessing   | sentiment_engine.py → clean_text()      |
| Stopword removal     | clean_text() — keeps negations          |
| Lemmatization        | clean_text() — watches → watch          |
| TF-IDF               | TfidfVectorizer with bigrams            |
| Cosine similarity    | Underlying LR classification            |
| VADER                | Lexicon-based sentiment scoring         |
| Precision/Recall/F1  | Printed after training in Step 3        |
| Real data retrieval  | scraper.py → SerpAPI Amazon integration |

---

## Quick demo without dataset (works immediately)

Skip Steps 2 and 3 — go straight to Step 5.
The dashboard will use VADER only + demo Amazon review data.
Still impressive for a demo!
