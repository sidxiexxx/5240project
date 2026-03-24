import streamlit as st
import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_sentiment_model():
    model_name = "sidxxxzzx/isom5240proj_amazon"  
    return pipeline("text-classification", model=model_name, return_all_scores=True)

@st.cache_resource
def load_zero_shot():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

sentiment_pipeline = load_sentiment_model()
zero_shot_pipeline = load_zero_shot()

labels = ["product quality", "delivery", "customer service", "price"]

# -----------------------------
# Helper Functions
# -----------------------------
def analyze_review(review):
    # Sentiment
    sentiment_result = sentiment_pipeline(review)[0]
    best = max(sentiment_result, key=lambda x: x['score'])

    label_map = {
        "LABEL_0": "Negative",
        "LABEL_1": "Positive",
        "NEGATIVE": "Negative",
        "POSITIVE": "Positive"
    }

    sentiment = label_map.get(best['label'], best['label'])
    confidence = round(best['score'], 2)

    return {
        "review": review,
        "sentiment": sentiment,
        "confidence": confidence,
    }

def extract_topics(review, candidate_labels):
    result = zero_shot_pipeline(review, candidate_labels)

    # 所有label + score
    topics = []
    for label, score in zip(result["labels"], result["scores"]):
        topics.append({
            "keyword": label,
            "score": round(score, 3)
        })

    return topics

def extract_topics_dict(review, candidate_labels):
    result = zero_shot_pipeline(review, candidate_labels)

    return {
        label: round(score, 3)
        for label, score in zip(result["labels"], result["scores"])
    }
# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🛒 E-commerce Customer Feedback Intelligence System")

# -------- Function 1: Text Input --------
st.header("Function 1: Single Review Analysis")
review_input = st.text_area("Enter a customer review:")
st.subheader("Custom Keywords")

user_input = st.text_input(
    "Enter keywords (comma separated):",
    "product quality, delivery, customer service, price"
)
candidate_labels = [x.strip() for x in user_input.split(",")]

if st.button("Analyze Review"):
    result = analyze_review(review_input)
    topics = extract_topics(review_input, candidate_labels)
    st.json(result)
    st.write("### Topics & Scores")
    st.json(topics)


# -------- Function 2: File Upload --------
st.header("Function 2: Batch Analysis (Upload Excel)")
uploaded_file = st.file_uploader("Upload Excel file with a 'review' column", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    if 'review' not in df.columns:
        st.error("Excel must contain a 'review' column")
    else:
        st.write("Preview:", df.head())

        if st.button("Run Batch Analysis"):
            results = []

            for review in df['review']:
                try:
                    res = analyze_review(str(review))
                    results.append(res)
                except Exception as e:
                    results.append({
                        "review": review,
                        "sentiment": "Error",
                        "confidence": 0,
                        "topic": "Error"
                    })

            result_df = pd.DataFrame(results)

            st.success("Analysis Complete!")
            st.dataframe(result_df)

            # Download button
            output_file = "analysis_results.xlsx"
            result_df.to_excel(output_file, index=False)

            with open(output_file, "rb") as f:
                st.download_button(
                    label="Download Results",
                    data=f,
                    file_name=output_file,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
