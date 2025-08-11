import os
import re
import pdfplumber
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util

# Function to extract text from PDF using pdfplumber
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text.strip()

# Function to clean and normalize text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)  # remove special chars
    text = re.sub(r'\s+', ' ', text)          # remove extra spaces
    return text.strip()

# Streamlit UI
st.title("AI-Powered Resume Screening Tool")

st.write("""
Upload multiple resumes (PDF files) and paste the job description below.  
The app will rank resumes based on how well they match the job description using semantic NLP.
""")

# Upload multiple resumes
uploaded_files = st.file_uploader("Upload Resume PDFs", type=["pdf"], accept_multiple_files=True)

# Input job description
job_description = st.text_area("Paste Job Description here", height=150)

# Load semantic model once
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

if st.button("Run Resume Screening"):

    if not uploaded_files:
        st.error("Please upload at least one resume PDF.")
    elif not job_description.strip():
        st.error("Please enter a job description.")
    else:
        # Process each uploaded resume
        resumes_data = []
        for uploaded_file in uploaded_files:
            text = extract_text_from_pdf(uploaded_file)
            cleaned_text = clean_text(text)
            resumes_data.append({
                "file_name": uploaded_file.name,
                "text": cleaned_text
            })

        df_resumes = pd.DataFrame(resumes_data)
        cleaned_jd = clean_text(job_description)

        # Encode texts using SentenceTransformer
        documents = [cleaned_jd] + df_resumes["text"].tolist()
        embeddings = model.encode(documents, convert_to_tensor=True)

        # Calculate cosine similarity with job description
        similarity_scores = util.cos_sim(embeddings[0], embeddings[1:]).cpu().numpy().flatten()

        df_resumes["Match Score"] = similarity_scores
        df_resumes = df_resumes.sort_values(by="Match Score", ascending=False)

        # Display ranked results
        st.subheader("Resume Ranking")
        st.write(f"Showing results for {len(df_resumes)} resumes:")

        for idx, row in df_resumes.iterrows():
            st.write(f"**{row['file_name']}** - Match Score: {row['Match Score']:.2f}")

        # Optional: downloadable CSV of results
        csv = df_resumes[["file_name", "Match Score"]].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Ranking as CSV",
            data=csv,
            file_name='resume_ranking.csv',
            mime='text/csv'
        )
