import PyPDF2

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Change 'resume1.pdf' to your actual PDF file name
pdf_file = "resume1.pdf"
resume_text = extract_text_from_pdf(pdf_file)

print("Extracted Text from PDF:")
print(resume_text)
