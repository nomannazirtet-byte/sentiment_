import streamlit as st
import pandas as pd
from transformers import pipeline
import numpy as np
import plotly.express as px

st.title("BERT-based Sentiment Analysis with Visualizations")

uploaded_file = st.file_uploader("Upload CSV with 'Review Text' column", type=["csv"])

if uploaded_file:
    # Read CSV (try UTF-16 encoding, fallback to default)
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-16', delimiter=',', on_bad_lines='skip')
    except:
        df = pd.read_csv(uploaded_file, on_bad_lines='skip')

    if "Review Text" not in df.columns:
        st.error("Please upload a CSV with a 'Review Text' column.")
        st.stop()

    # Load Hugging Face sentiment pipeline
    sentiment_pipeline = pipeline("sentiment-analysis")

    # Clean inputs: convert all to string and replace NaN with empty string
    texts = df['Review Text'].fillna("").astype(str).tolist()

    st.write("Predicting sentiments using BERT model...")
    predictions = sentiment_pipeline(texts)

    # Add sentiment and confidence to DataFrame
    df['Sentiment'] = [pred['label'] for pred in predictions]
    df['Confidence'] = [pred['score'] for pred in predictions]

    # Show preview
    st.subheader("🔍 Sample of Sentiment Predictions")
    st.dataframe(df[['Review Text', 'Sentiment', 'Confidence']].head())

    # --- VISUALIZATIONS ---

    st.subheader("📊 Sentiment Distribution")

    sentiment_counts = df['Sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']

    fig_sentiment = px.pie(
        sentiment_counts,
        names='Sentiment',
        values='Count',
        color='Sentiment',
        color_discrete_map={"NEGATIVE": "red", "NEUTRAL": "gray", "POSITIVE": "green"},
        title='Sentiment Distribution'
    )
    st.plotly_chart(fig_sentiment, use_container_width=True)

    # Sentiment vs Star Rating (if available)
    if 'Star Rating' in df.columns:
        st.subheader("⭐ Sentiment vs Star Rating")

        star_sentiment = df.groupby(['Star Rating', 'Sentiment']).size().reset_index(name='Count')

        fig_star = px.bar(
            star_sentiment,
            x='Star Rating',
            y='Count',
            color='Sentiment',
            barmode='group',
            category_orders={"Sentiment": ["NEGATIVE", "NEUTRAL", "POSITIVE"]},
            color_discrete_map={"NEGATIVE": "red", "NEUTRAL": "gray", "POSITIVE": "green"},
            title="Sentiment by Star Rating"
        )
        st.plotly_chart(fig_star, use_container_width=True)

    # Confidence distribution
    st.subheader("🔎 Confidence Score Distribution")

    fig_conf = px.histogram(
        df,
        x='Confidence',
        nbins=20,
        title='Distribution of Sentiment Prediction Confidence'
    )
    st.plotly_chart(fig_conf, use_container_width=True)

    # Device vs Sentiment plot
    if 'Device' in df.columns:
        st.subheader("📱 Sentiment Distribution by Device")

        device_sentiment = df.groupby(['Device', 'Sentiment']).size().reset_index(name='Count')
        top_devices = df['Device'].value_counts().nlargest(10).index
        device_sentiment = device_sentiment[device_sentiment['Device'].isin(top_devices)]

        fig_device = px.bar(
            device_sentiment,
            x='Device',
            y='Count',
            color='Sentiment',
            barmode='group',
            category_orders={"Sentiment": ["NEGATIVE", "NEUTRAL", "POSITIVE"]},
            color_discrete_map={"NEGATIVE": "red", "NEUTRAL": "gray", "POSITIVE": "green"},
            title="Top 10 Devices by Sentiment Distribution"
        )
        st.plotly_chart(fig_device, use_container_width=True)

    # Download option
    st.subheader("📥 Download Results")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV with Sentiment",
        data=csv,
        file_name="bert_sentiment_output.csv",
        mime="text/csv"
    )
