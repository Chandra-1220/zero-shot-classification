# app.py

import streamlit as st
from transformers import pipeline

# ---- Page Config ----
st.set_page_config(page_title="Zero-Shot Text Classifier", page_icon="🔍", layout="centered")

st.title("🔍 Zero-Shot Text Classification")
st.write("Classify text into any custom categories — no training required.")

# ---- Load Model (cached so it doesn't reload on every interaction) ----
@st.cache_resource
def load_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

classifier = load_classifier()

# ---- User Inputs ----
sequence = st.text_area("Enter the text to classify:", height=150,
                         placeholder="e.g. The new smartphone has an amazing camera and battery life.")

labels_input = st.text_input("Enter candidate labels (comma-separated):",
                              placeholder="e.g. technology, sports, politics, food")

multi_label = st.checkbox("Allow multiple labels to apply (multi-label)", value=False)

# ---- Run Classification ----
if st.button("Classify"):
    if not sequence.strip():
        st.warning("Please enter some text to classify.")
    elif not labels_input.strip():
        st.warning("Please enter at least one candidate label.")
    else:
        candidate_labels = [label.strip() for label in labels_input.split(",") if label.strip()]

        with st.spinner("Classifying..."):
            result = classifier(sequence, candidate_labels, multi_label=multi_label)

        st.subheader("Results")
        st.write(f"**Text:** {result['sequence']}")

        for label, score in zip(result['labels'], result['scores']):
            st.write(f"**{label}**")
            st.progress(float(score))
            st.write(f"{score:.4f}")
            st.markdown("---")

        top_label = result['labels'][0]
        top_score = result['scores'][0]
        st.success(f"🏆 Top prediction: **{top_label}** ({top_score:.2%} confidence)")
