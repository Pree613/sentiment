import streamlit as st
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import io
from wordcloud import WordCloud
import nltk

# Download NLTK resources
nltk.download("vader_lexicon")

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Custom CSS Styling
st.markdown(
    """
    <style>
    .css-18e3th9 { padding-top: 50px; }
    .stApp { background-color: #f4f4f4; }
    </style>
    """,
    unsafe_allow_html=True
)

# App Title & Description
st.markdown("<h1 style='text-align: center; color: blue;'>Sentiment Analysis Dashboard</h1>", unsafe_allow_html=True)
st.markdown("Analyze text sentiment trends with **interactive insights & NLP models (TextBlob & Vader)**")

# Sidebar Filters
st.sidebar.header("Filter Options")
analysis_type = st.sidebar.radio("Choose Analysis:", ["Text Input", "CSV File Upload"])
sentiment_filter = st.sidebar.selectbox("Filter by Sentiment:", ["All", "Positive 😊", "Neutral 😐", "Negative 😡"])
date_filter = st.sidebar.date_input("Select Start Date", None)

# Function to analyze sentiment
def analyze_sentiment(text):
    """Performs sentiment analysis using TextBlob and Vader, returning sentiment label & score."""
    if not text.strip():
        return "Neutral 😐", 0.0
    
    blob_sentiment = TextBlob(text).sentiment.polarity
    vader_sentiment = sia.polarity_scores(text)["compound"]
    final_score = (blob_sentiment + vader_sentiment) / 2

    if final_score > 0:
        return "Positive 😊", final_score
    elif final_score < 0:
        return "Negative 😡", final_score
    else:
        return "Neutral 😐", final_score

# Text Input for real-time analysis
if analysis_type == "Text Input":
    user_text = st.text_area("Enter text for analysis:")
    if user_text:
        sentiment, score = analyze_sentiment(user_text)
        st.subheader("Sentiment Result:")
        st.markdown(f"**{sentiment}** (Score: {score:.2f})")

        # Generate Word Cloud
        wordcloud = WordCloud(width=400, height=200, background_color="white").generate(user_text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

# CSV File Upload for Batch Analysis
if analysis_type == "CSV File Upload":
    uploaded_file = st.file_uploader("Upload a CSV file (must have 'text' column)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file, on_bad_lines="skip", engine="python")
        df.columns = df.columns.str.strip()

        if "text" in df.columns:
            df["Sentiment"], df["Score"] = zip(*df["text"].astype(str).apply(analyze_sentiment))

            # Handle "date" column if available
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"])

            # Apply sentiment filter
            if sentiment_filter != "All":
                df = df[df["Sentiment"] == sentiment_filter]

            # Apply date filter
            if date_filter and "date" in df.columns:
                df = df[df["date"] >= pd.to_datetime(date_filter)]

            st.subheader("📊 Sentiment Analysis Results")
            display_columns = ["text", "Sentiment", "Score"]
            if "date" in df.columns:
                display_columns.insert(0, "date")
            st.dataframe(df[display_columns], height=300)

            # Sentiment Trend Analysis (Line Chart)
            if "date" in df.columns and not df.empty:
                daily_sentiment = df.groupby(df["date"].dt.date)["Score"].mean().reset_index()
                fig_trend = px.line(daily_sentiment, x="date", y="Score", title="Sentiment Trend Over Time")
                st.plotly_chart(fig_trend)

            # Sentiment Distribution (Bar & Pie Chart)
            if not df.empty:
                sentiment_counts = df["Sentiment"].value_counts()
                fig_bar = px.bar(sentiment_counts, x=sentiment_counts.index, y=sentiment_counts.values, 
                                 color=sentiment_counts.index, title="Sentiment Distribution")
                st.plotly_chart(fig_bar)

                fig_pie = px.pie(df, names="Sentiment", title="Sentiment Proportion")
                st.plotly_chart(fig_pie)

            # Generate Word Clouds for Each Sentiment
            st.subheader("Word Clouds by Sentiment")
            for sentiment in ["Positive 😊", "Negative 😡", "Neutral 😐"]:
                words = " ".join(df[df["Sentiment"] == sentiment]["text"].dropna().astype(str))
                if words.strip():
                    wordcloud = WordCloud(width=400, height=200, background_color="white").generate(words)
                    fig, ax = plt.subplots()
                    ax.imshow(wordcloud, interpolation="bilinear")
                    ax.axis("off")
                    st.markdown(f"**{sentiment} Words**")
                    st.pyplot(fig)
                else:
                    st.warning(f"No text available for {sentiment} Word Cloud.")

            # Download Button for Processed Data
            buffer = io.BytesIO()
            df.to_csv(buffer, index=False)
            buffer.seek(0)
            st.download_button(label="Download Results CSV", data=buffer, file_name="sentiment_analysis.csv", mime="text/csv")
        else:
            st.error("CSV file must contain a 'text' column.")

# Footer
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit")
