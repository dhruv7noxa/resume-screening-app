import os
import PyPDF2
import pandas as pd

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text.strip()

# Folder containing resumes (current folder in this case)
folder_path = "."

# List to store resume data
resumes_data = []

for file_name in os.listdir(folder_path):
    if file_name.lower().endswith(".pdf"):
        resume_text = extract_text_from_pdf(os.path.join(folder_path, file_name))
        resumes_data.append({"file_name": file_name, "text": resume_text})

# Convert to DataFrame
df_resumes = pd.DataFrame(resumes_data)

print("Loaded Resumes:\n", df_resumes[["file_name"]])
print("\nSample Extracted Text from First Resume:\n", df_resumes.iloc[0]["text"][:500])  # show first 500 chars
