import streamlit as st
import pandas as pd
import re
from translate import Translator
import joblib
import numpy as np
import plotly.express as px
from datetime import timedelta

# Load trained model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

translator = Translator(to_lang="en")

def translate_to_english(text):
    try:
        if pd.isnull(text):
            return ""
        return translator.translate(str(text))
    except:
        return text

def clean_text(text):
    if pd.isnull(text):
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

def is_ad_related(review):
    ad_keywords = ['ad', 'ads', 'advertisement', 'advertisements', 'commercial', 'sponsored']
    if pd.isnull(review):
        return False
    review = str(review).lower()
    return any(keyword in review for keyword in ad_keywords)

st.title("📑 Bulk Sentiment Analysis with Advanced Visualization")

uploaded_file = st.file_uploader("Upload CSV with 'Review Text' column", type=["csv"])

if uploaded_file:
    try:
        # Read CSV with possible UTF-16 and tab delimiter fallback
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-16', delimiter=',', on_bad_lines='skip')
        except:
            df = pd.read_csv(uploaded_file, on_bad_lines='skip')

        # Clean up column names
        df.columns = df.columns.str.replace(r'[^\x00-\x7F]+', '', regex=True).str.strip()

        # Rename columns for ease of use
        rename_map = {
            'App Version Name': 'App Version',
            'Reviewer Language': 'Language',
            'Review Submit Date and Time': 'Review Date',
            'Developer Reply Date and Time': 'Reply Date',
            'Developer Reply Text': 'Developer Reply'
        }
        df.rename(columns=rename_map, inplace=True)

        # Check mandatory column
        if "Review Text" not in df.columns:
            st.error("The uploaded CSV must have a column named 'Review Text'.")
            st.stop()

        # Convert types
        df['Star Rating'] = pd.to_numeric(df['Star Rating'], errors='coerce')
        df['Review Date'] = pd.to_datetime(df['Review Date'], errors='coerce').dt.tz_localize(None)
        if 'Reply Date' in df.columns:
            df['Reply Date'] = pd.to_datetime(df['Reply Date'], errors='coerce').dt.tz_localize(None)

        # Drop rows with empty/missing Review Text
        df = df.dropna(subset=["Review Text"])

        st.success("✅ CSV loaded and preprocessed successfully!")

        # Show original sample
        st.subheader("📋 Original Sample")
        st.write(df[['Review Text', 'Star Rating', 'App Version', 'Language', 'Device', 'Review Date']].head())

        # Preprocess reviews
        st.subheader("⚙️ Processing reviews...")
        df['Translated_Review'] = df['Review Text'].apply(translate_to_english)
        df['Is_Ad'] = df['Translated_Review'].apply(is_ad_related)
        df = df[~df['Is_Ad']]  # Remove ads
        df['Cleaned_Review'] = df['Translated_Review'].apply(clean_text)

        # Predict sentiment
        vecs = vectorizer.transform(df['Cleaned_Review'])
        predictions = model.predict(vecs)
        probabilities = model.predict_proba(vecs)

        df['Sentiment'] = [label_map[p] for p in predictions]
        df['Confidence'] = np.max(probabilities, axis=1)
        df['Sentiment_Score'] = predictions  # For numeric sentiment score

        # === KEY METRICS ===
        st.subheader("🔹 Key Metrics")

        total_reviews = len(df)
        pos_pct = (df['Sentiment'] == 'Positive').mean() * 100
        neu_pct = (df['Sentiment'] == 'Neutral').mean() * 100
        neg_pct = (df['Sentiment'] == 'Negative').mean() * 100
        avg_star = df['Star Rating'].mean()
        avg_sentiment_score = df['Sentiment_Score'].mean()
        num_dev_reply = df['Developer Reply'].dropna().astype(bool).sum()
        pct_dev_reply = num_dev_reply / total_reviews * 100 if total_reviews > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Reviews", total_reviews)
        col2.metric("% Positive", f"{pos_pct:.1f}%")
        col3.metric("% Neutral", f"{neu_pct:.1f}%")
        col1.metric("% Negative", f"{neg_pct:.1f}%")
        col2.metric("Average Star Rating", f"{avg_star:.2f}" if not np.isnan(avg_star) else "N/A")
        col3.metric("Avg Sentiment Score", f"{avg_sentiment_score:.2f}" if not np.isnan(avg_sentiment_score) else "N/A")
        col1.metric("Reviews with Developer Reply", f"{num_dev_reply} ({pct_dev_reply:.1f}%)")

        # === SENTIMENT DISTRIBUTION PIE/DOUGHNUT CHART ===
        st.subheader("🔹 Overall Sentiment Distribution")
        sentiment_counts = df['Sentiment'].value_counts().reindex(['Negative', 'Neutral', 'Positive'], fill_value=0)
        fig_pie = px.pie(
            names=sentiment_counts.index,
            values=sentiment_counts.values,
            title="Sentiment Distribution",
            color=sentiment_counts.index,
            color_discrete_map={"Negative":"red", "Neutral":"gray", "Positive":"green"},
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # === SENTIMENT BY APP VERSION BAR CHART ===
        if 'App Version' in df.columns:
            st.subheader("🔹 Sentiment by App Version")
            sent_app_ver = df.groupby(['App Version', 'Sentiment']).size().reset_index(name='Count')
            fig_bar = px.bar(
                sent_app_ver,
                x='App Version',
                y='Count',
                color='Sentiment',
                category_orders={'Sentiment': ['Negative', 'Neutral', 'Positive']},
                color_discrete_map={"Negative":"red", "Neutral":"gray", "Positive":"green"},
                title="Sentiment Distribution by App Version"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # === STACKED BAR: Star Rating vs Sentiment ===
        if 'Star Rating' in df.columns:
            st.subheader("🔹 Star Rating vs Sentiment")
            # Drop NaN star rating for this chart
            df_star_sent = df.dropna(subset=['Star Rating'])
            star_sent_counts = df_star_sent.groupby(['Star Rating', 'Sentiment']).size().reset_index(name='Count')
            fig_stack = px.bar(
                star_sent_counts,
                x='Star Rating',
                y='Count',
                color='Sentiment',
                category_orders={'Sentiment': ['Negative', 'Neutral', 'Positive']},
                color_discrete_map={"Negative":"red", "Neutral":"gray", "Positive":"green"},
                title="Star Rating vs Sentiment"
            )
            st.plotly_chart(fig_stack, use_container_width=True)

        # === TIME TRENDS ===
        st.subheader("🔹 Time Trends")
        if 'Review Date' in df.columns:
            # Reviews over time (weekly)
            df_time = df.dropna(subset=['Review Date'])
            df_time['Week'] = df_time['Review Date'].dt.to_period('W').apply(lambda r: r.start_time)
            reviews_weekly = df_time.groupby('Week').size().reset_index(name='Review Count')
            fig_line_reviews = px.line(
                reviews_weekly,
                x='Week',
                y='Review Count',
                title='Number of Reviews Over Time (Weekly)'
            )
            st.plotly_chart(fig_line_reviews, use_container_width=True)

            # Sentiment trend over time (% Positive vs Negative)
            sent_time = df_time.groupby(['Week', 'Sentiment']).size().unstack(fill_value=0)
            sent_time['Total'] = sent_time.sum(axis=1)
            sent_time['% Positive'] = sent_time.get('Positive', 0) / sent_time['Total'] * 100
            sent_time['% Negative'] = sent_time.get('Negative', 0) / sent_time['Total'] * 100
            fig_sent_trend = px.line(
                sent_time.reset_index(),
                x='Week',
                y=['% Positive', '% Negative'],
                labels={'value':'Percentage', 'Week':'Week'},
                title='Sentiment Trend Over Time'
            )
            st.plotly_chart(fig_sent_trend, use_container_width=True)

            # Average star rating by month
            df_time['Month'] = df_time['Review Date'].dt.to_period('M').apply(lambda r: r.start_time)
            avg_star_month = df_time.groupby('Month')['Star Rating'].mean().reset_index()
            fig_star_month = px.bar(
                avg_star_month,
                x='Month',
                y='Star Rating',
                title='Average Star Rating by Month'
            )
            st.plotly_chart(fig_star_month, use_container_width=True)

        # === DRILL-DOWN ANALYSIS ===
        st.subheader("🔹 Drill-Down Analysis")
        # Prepare clickable links if available
        if 'Review Link' in df.columns:
            df['Review Link'] = df['Review Link'].fillna('')
            df['Review Link'] = df['Review Link'].apply(lambda x: f"[Link]({x})" if x else "")

        # Select columns to show (adjust as needed)
        show_cols = ['Review Text', 'Sentiment', 'Star Rating', 'App Version', 'Device', 'Language', 'Review Date']
        if 'Review Link' in df.columns:
            show_cols.append('Review Link')
        if 'Developer Reply' in df.columns:
            show_cols.append('Developer Reply')

        # Show data table with markdown for links
        def render_markdown_links(df_in):
            # Streamlit dataframe does not render markdown in cells, so render manually
            for i, row in df_in.iterrows():
                cols = []
                for c in show_cols:
                    val = row[c]
                    if c == 'Review Link' and val:
                        cols.append(val)
                    else:
                        cols.append(str(val))
                st.markdown(" | ".join(cols))

        # Instead of render_markdown_links, show a simpler table:
        # For clickable links in Streamlit's dataframe, we can use st.markdown with unsafe_allow_html=True per cell,
        # but that is complicated. So we just show normal table here.
        st.dataframe(df[show_cols].fillna(''))

        # === DEVELOPER REPLY ANALYSIS ===
        st.subheader("🔹 Developer Reply Analysis")
        if 'Developer Reply' in df.columns:
            num_replied = df['Developer Reply'].dropna().astype(bool).sum()
            pct_replied = num_replied / total_reviews * 100 if total_reviews > 0 else 0

            st.write(f"Reviews with developer replies: {num_replied} ({pct_replied:.1f}%)")

            # Calculate average response time if dates available
            if 'Reply Date' in df.columns:
                df_reply = df.dropna(subset=['Review Date', 'Reply Date']).copy()
                df_reply['Response Time'] = (df_reply['Reply Date'] - df_reply['Review Date']).dt.total_seconds() / 3600  # hours
                avg_response_time = df_reply['Response Time'].mean()
                st.write(f"Average response time: {avg_response_time:.1f} hours")

            # Bar chart: Sentiment of reviews with reply vs no reply
            df['Has Reply'] = df['Developer Reply'].notna()
            sent_reply = df.groupby(['Has Reply', 'Sentiment']).size().reset_index(name='Count')
            fig_reply_sent = px.bar(
                sent_reply,
                x='Has Reply',
                y='Count',
                color='Sentiment',
                category_orders={'Sentiment': ['Negative', 'Neutral', 'Positive']},
                color_discrete_map={"Negative":"red", "Neutral":"gray", "Positive":"green"},
                title="Sentiment of Reviews with/without Developer Reply",
                labels={'Has Reply': 'Has Developer Reply'}
            )
            st.plotly_chart(fig_reply_sent, use_container_width=True)

            # === STACKED BAR: Star Rating vs Sentiment ===
            # === STACKED BAR: Star Rating vs Sentiment ===
            if 'Star Rating' in df.columns:
                st.subheader("🔹 Star Rating vs Sentiment")
                df_star_sent = df.dropna(subset=['Star Rating'])
                star_sent_counts = df_star_sent.groupby(['Star Rating', 'Sentiment']).size().reset_index(name='Count')
                fig_stack = px.bar(
                    star_sent_counts,
                    x='Star Rating',
                    y='Count',
                    color='Sentiment',
                    category_orders={'Sentiment': ['Negative', 'Neutral', 'Positive']},
                    color_discrete_map={"Negative": "red", "Neutral": "gray", "Positive": "green"},
                    title="Star Rating vs Sentiment"
                )
                st.plotly_chart(fig_stack, use_container_width=True, key="star_rating_vs_sentiment")

                # === BAR CHART: Star Rating Counts ===
                st.subheader("🔹 Star Rating Distribution")
                star_counts = df['Star Rating'].value_counts().sort_index()
                fig_star_dist = px.bar(
                    x=star_counts.index.astype(str),
                    y=star_counts.values,
                    labels={'x': 'Star Rating', 'y': 'Number of Reviews'},
                    title="Number of Reviews by Star Rating"
                )
                fig_star_dist.update_traces(text=star_counts.values, textposition='outside')
                st.plotly_chart(fig_star_dist, use_container_width=True, key="star_rating_distribution")



    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
