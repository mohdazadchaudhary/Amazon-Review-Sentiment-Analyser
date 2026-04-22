# 🚀 Amazon Review Sentiment Analyser

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io/)
[![NLP](https://img.shields.io/badge/NLP-VADER%20%2B%20TF--IDF-green.svg)](https://en.wikipedia.org/wiki/Natural_language_processing)
[![Machine Learning](https://img.shields.io/badge/ML-Logistic%20Regression-orange.svg)](https://scikit-learn.org/)

An advanced **End-to-End Sentiment Analysis Dashboard** that fetches real-time Amazon product reviews and performs deep sentiment classification using hybrid NLP techniques.

---

## 📸 Dashboard Preview

![Main Dashboard](assets/Screenshot%202026-04-22%20204254.png)

---

## ✨ Key Features

- **🔍 Real-time Scraping:** Fetch live reviews from any Amazon product using ASIN or URL (via SerpAPI).
- **🧠 Hybrid AI Model:** Combines Lexicon-based (VADER) and Machine Learning (Logistic Regression) for maximum accuracy.
- **📊 Visual Analytics:** 
  - Sentiment Distribution (Donut Charts)
  - Star Rating vs. Sentiment Correlation
  - Interactive Word Clouds
- **⚖️ Model Comparison:** Real-time agreement rate between VADER and Logistic Regression.
- **📂 Review Breakdown:** Filter and read through reviews categorized by their sentiment.

---

## 🛠️ Getting Started

### 1️⃣ Installation
Open your terminal in the project directory and run:
```bash
pip install -r requirements.txt
```

### 2️⃣ Dataset Preparation (Optional)
To train the model on the full IMDB dataset:
1. Download the [IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).
2. Place `IMDB Dataset.csv` in the root folder.
3. Run the trainer:
   ```bash
   python sentiment_engine.py
   ```

### 3️⃣ Real Amazon Data (Optional)
To fetch live reviews instead of demo data:
1. Get a free API key from [SerpAPI](https://serpapi.com).
2. Open `scraper.py` and replace `YOUR_SERPAPI_KEY_HERE` with your key.

### 4️⃣ Launch the Dashboard
```bash
streamlit run app.py
```

---

## 📚 IRTM Concepts Used

This project implements core Information Retrieval and Text Mining (IRTM) principles:

| Concept | Implementation |
| :--- | :--- |
| **Text Preprocessing** | Cleaning, tokenization, and noise reduction |
| **Lemmatization** | Word normalization using NLTK |
| **TF-IDF Vectorization** | Converting text to numerical features with Bigrams |
| **Sentiment Lexicons** | VADER-based rule analysis |
| **Supervised Learning** | Logistic Regression classification |
| **Performance Metrics** | Precision, Recall, and F1-Score evaluation |

---

## 🖼️ Visual Gallery

<p align="center">
  <img src="assets/Screenshot%202026-04-22%20204307.png" width="45%" />
  <img src="assets/Screenshot%202026-04-22%20204334.png" width="45%" />
</p>
<p align="center">
  <img src="assets/Screenshot%202026-04-22%20204346.png" width="45%" />
  <img src="assets/Screenshot%202026-04-22%20204355.png" width="45%" />
</p>

---

## 📁 File Structure

- `app.py`: The Streamlit frontend dashboard.
- `sentiment_engine.py`: ML logic, preprocessing, and model training.
- `scraper.py`: Amazon scraping logic via SerpAPI.
- `requirements.txt`: Project dependencies.
- `assets/`: UI screenshots and visual assets.

---

<p align="center">Made for IRTM Mini Project</p>
