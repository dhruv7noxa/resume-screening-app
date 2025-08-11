import os
import re
import pdfplumber
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text.strip()

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)  # remove special chars
    text = re.sub(r'\s+', ' ', text)          # remove extra spaces
    return text.strip()

# Load resumes from PDFs in current folder
folder_path = "."
resumes_data = []
for file_name in os.listdir(folder_path):
    if file_name.lower().endswith(".pdf"):
        resume_text = extract_text_from_pdf(os.path.join(folder_path, file_name))
        resumes_data.append({"file_name": file_name, "text": clean_text(resume_text)})

df_resumes = pd.DataFrame(resumes_data)

# Load and clean job description
with open("job_description.txt", "r", encoding="utf-8") as f:
    job_description = clean_text(f.read())

# Load semantic model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode JD and all resumes into embeddings
embeddings = model.encode([job_description] + df_resumes["text"].tolist(), convert_to_tensor=True)

# Calculate cosine similarity between JD (first embedding) and each resume
similarity_scores = util.cos_sim(embeddings[0], embeddings[1:]).cpu().numpy().flatten()

# Add scores to DataFrame and sort
df_resumes["similarity"] = similarity_scores
df_resumes = df_resumes.sort_values(by="similarity", ascending=False)

# Show rankings
print("\nResume Ranking based on Job Description match (Semantic Matching):\n")
for idx, row in df_resumes.iterrows():
    print(f"{row['file_name']} - Match Score: {row['similarity']:.2f}")
