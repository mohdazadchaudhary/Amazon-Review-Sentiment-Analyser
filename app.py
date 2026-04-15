"""
app.py  —  Sentiment Analysis Dashboard
=========================================
Run with:  streamlit run app.py

Features:
  - Paste Amazon URL or ASIN to fetch real reviews
  - Analyse with VADER + Logistic Regression
  - Sentiment distribution chart
  - Star rating vs sentiment comparison
  - Review-by-review breakdown
  - Top positive / negative words
  - Live text input to test any sentence
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from scraper import fetch_reviews, extract_asin
from sentiment_engine import load_model, predict, top_features


# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Review Sentiment Analyser",
    page_icon="🔍",
    layout="wide"
)

st.title("Amazon Review Sentiment Analyser")
st.caption("IRTM Mini Project  |  VADER + Logistic Regression  |  Real Amazon Data")

# ── Load model ─────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    return load_model()

model, vectorizer = get_model()

if model is None:
    st.warning(
        "Logistic Regression model not found. "
        "Run `python sentiment_engine.py` first to train it. "
        "VADER analysis will still work."
    )

# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    mode = st.radio(
        "Analysis mode",
        ["Amazon product (ASIN/URL)", "Type your own review"]
    )

    st.divider()
    st.header("About")
    st.markdown("""
**IRTM Concepts used:**
- Text preprocessing
- TF-IDF vectorization
- Logistic Regression
- VADER lexicon analysis
- Precision / Recall / F1

**Free APIs:**
- SerpAPI (Amazon reviews)
- No paid services needed
    """)

    if model:
        st.success("Model loaded ✓")
    else:
        st.error("Model not trained yet")


# ══════════════════════════════════════════════════════════════════════
# MODE 1 — Amazon Product Analysis
# ══════════════════════════════════════════════════════════════════════
if mode == "Amazon product (ASIN/URL)":

    st.subheader("Analyse Amazon Product Reviews")

    col1, col2 = st.columns([3, 1])
    with col1:
        user_input = st.text_input(
            "Paste Amazon product URL or ASIN",
            placeholder="e.g.  B09G9FPHY6  or  https://www.amazon.in/dp/B09G9FPHY6"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyse_btn = st.button("Analyse Reviews", type="primary", use_container_width=True)

    # Demo tip
   # st.caption("No API key yet? Leave the SERPAPI_KEY as-is in scraper.py — demo data will be used automatically.")

    if analyse_btn and user_input:

        asin = extract_asin(user_input)

        if not asin:
            st.error("Could not extract ASIN. Please check the URL or ASIN format.")
            st.stop()

        with st.spinner(f"Fetching reviews for ASIN: {asin} ..."):
            data = fetch_reviews(asin, num_pages=2)

        if data['error']:
            st.error(f"Error fetching reviews: {data['error']}")
            st.stop()

        reviews = data['reviews']
        if not reviews:
            st.warning("No reviews found for this product.")
            st.stop()

        # ── Run sentiment on each review ──
        results = []
        for r in reviews:
            full_text = r['title'] + ' ' + r['text']
            sentiment = predict(full_text, model, vectorizer)
            results.append({
                'title'       : r['title'],
                'text'        : r['text'],
                'star_rating' : r['rating'],
                'date'        : r['date'],
                'vader_label' : sentiment['vader']['label'],
                'vader_score' : sentiment['vader']['compound'],
                'lr_label'    : sentiment['logistic_regression']['label'] if sentiment['logistic_regression'] else 'N/A',
                'lr_pos_pct'  : sentiment['logistic_regression']['pos_pct'] if sentiment['logistic_regression'] else 0,
                'final_label' : sentiment['final_label'],
            })

        df = pd.DataFrame(results)

        # ── Product header ──
        st.divider()
        st.subheader(data.get('product_name', asin))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Reviews analysed",  len(df))
        m2.metric("Avg star rating",   data.get('rating', 'N/A'))
        pos_count = (df['final_label'] == 'Positive').sum()
        neg_count = (df['final_label'] == 'Negative').sum()
        m3.metric("Positive reviews",  f"{pos_count} ({round(pos_count/len(df)*100)}%)")
        m4.metric("Negative reviews",  f"{neg_count} ({round(neg_count/len(df)*100)}%)")

        # ── Charts row ──
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Sentiment distribution")
            counts = df['final_label'].value_counts().reset_index()
            counts.columns = ['Sentiment', 'Count']
            color_map = {'Positive': '#1D9E75', 'Negative': '#E24B4A', 'Neutral': '#888780'}
            fig = px.pie(
                counts, names='Sentiment', values='Count',
                color='Sentiment', color_discrete_map=color_map,
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Star rating vs sentiment")
            star_sent = df.groupby('star_rating')['final_label'].apply(
                lambda x: (x == 'Positive').sum() / len(x) * 100
            ).reset_index()
            star_sent.columns = ['Stars', 'Positive %']
            fig2 = px.bar(
                star_sent, x='Stars', y='Positive %',
                color='Positive %',
                color_continuous_scale=['#E24B4A', '#FAEEDA', '#1D9E75'],
                range_color=[0, 100],
                labels={'Stars': 'Star Rating', 'Positive %': '% Positive Reviews'}
            )
            fig2.update_layout(
                coloraxis_showscale=False,
                margin=dict(t=10, b=10, l=10, r=10)
            )
            st.plotly_chart(fig2, use_container_width=True)

        # ── VADER vs LR comparison ──
        if model:
            st.subheader("VADER vs Logistic Regression comparison")
            agree    = (df['vader_label'] == df['lr_label']).sum()
            disagree = len(df) - agree
            ca, cb, cc = st.columns(3)
            ca.metric("Both agree",    agree)
            cb.metric("Disagree",      disagree)
            cc.metric("Agreement rate", f"{round(agree/len(df)*100)}%")
            st.caption("When models disagree, Logistic Regression is used as the final verdict (it's more accurate).")

        # ── Word cloud ──
        st.subheader("Word cloud — all reviews")
        all_text = ' '.join(df['title'] + ' ' + df['text'])
        try:
            wc = WordCloud(
                width=800, height=300,
                background_color='white',
                colormap='RdYlGn',
                max_words=80
            ).generate(all_text)
            fig_wc, ax = plt.subplots(figsize=(10, 3))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig_wc)
        except Exception:
            st.info("Install wordcloud for word cloud: pip install wordcloud")

        # ── Top features (if model loaded) ──
        if model:
            st.subheader("What the model learned — top sentiment words")
            features = top_features(model, vectorizer, n=8)
            fw1, fw2 = st.columns(2)
            with fw1:
                st.markdown("**Top Positive words**")
                pos_df = pd.DataFrame(features['positive_words'], columns=['Word', 'Weight'])
                st.dataframe(pos_df, hide_index=True, use_container_width=True)
            with fw2:
                st.markdown("**Top Negative words**")
                neg_df = pd.DataFrame(features['negative_words'], columns=['Word', 'Weight'])
                st.dataframe(neg_df, hide_index=True, use_container_width=True)

        # ── Individual reviews ──
        st.subheader("Review breakdown")
        filter_col, _ = st.columns([2, 3])
        with filter_col:
            sentiment_filter = st.selectbox(
                "Filter by sentiment", ["All", "Positive", "Negative", "Neutral"]
            )

        filtered_df = df if sentiment_filter == "All" else df[df['final_label'] == sentiment_filter]

        for _, row in filtered_df.iterrows():
            color = "#1D9E75" if row['final_label'] == 'Positive' else \
                    "#E24B4A" if row['final_label'] == 'Negative' else "#888780"
            with st.container():
                st.markdown(
                    f"""<div style="border-left: 4px solid {color};
                                    padding: 8px 12px;
                                    margin-bottom: 8px;
                                    border-radius: 0 8px 8px 0;
                                    background: #fafafa">
                        <b>{row['title']}</b>
                        <span style="color:{color};font-size:12px;margin-left:8px">
                            {row['final_label']}
                        </span>
                        <span style="color:#888;font-size:12px;margin-left:8px">
                            {'★' * int(row['star_rating'])} {row['date']}
                        </span>
                        <p style="margin:4px 0 0;font-size:13px;color:#444">{row['text']}</p>
                    </div>""",
                    unsafe_allow_html=True
                )


# ══════════════════════════════════════════════════════════════════════
# MODE 2 — Live Text Input
# ══════════════════════════════════════════════════════════════════════
else:
    st.subheader("Test any review text live")
    st.caption("Type or paste any sentence — see both models analyse it in real time.")

    user_text = st.text_area(
        "Enter review text",
        placeholder="e.g. This watch is absolutely brilliant! Superb quality and great value.",
        height=120
    )

    if user_text.strip():
        result = predict(user_text, model, vectorizer)

        v = result['vader']
        lr = result['logistic_regression']

        # Big verdict
        final = result['final_label']
        verdict_color = "#1D9E75" if final == 'Positive' else \
                        "#E24B4A" if final == 'Negative' else "#888780"
        st.markdown(
            f"""<div style="text-align:center;padding:20px;
                            border-radius:12px;background:{verdict_color}20;
                            border:2px solid {verdict_color};margin:16px 0">
                <h2 style="color:{verdict_color};margin:0">{final}</h2>
                <p style="color:{verdict_color};margin:4px 0 0;font-size:14px">Final verdict</p>
            </div>""",
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### VADER (lexicon-based)")
            st.metric("Compound score", v['compound'])
            fig = go.Figure(go.Bar(
                x=[v['pos_pct'], v['neg_pct'], v['neu_pct']],
                y=['Positive', 'Negative', 'Neutral'],
                orientation='h',
                marker_color=['#1D9E75', '#E24B4A', '#888780']
            ))
            fig.update_layout(
                xaxis_title="Percentage",
                margin=dict(t=10, b=10, l=10, r=10),
                height=180
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if lr:
                st.markdown("#### Logistic Regression (trained)")
                st.metric("Positive confidence", f"{lr['pos_pct']}%")
                fig2 = go.Figure(go.Bar(
                    x=[lr['pos_pct'], lr['neg_pct']],
                    y=['Positive', 'Negative'],
                    orientation='h',
                    marker_color=['#1D9E75', '#E24B4A']
                ))
                fig2.update_layout(
                    xaxis_title="Probability %",
                    margin=dict(t=10, b=10, l=10, r=10),
                    height=180
                )
                st.plotly_chart(fig2, use_container_width=True)
                agree = result.get('agreement')
                if agree is True:
                    st.success("Both models agree")
                elif agree is False:
                    st.warning("Models disagree — LR verdict used as final")
            else:
                st.info("Train model for Logistic Regression results:\npython sentiment_engine.py")
