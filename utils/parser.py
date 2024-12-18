import PyPDF2
import os
import sys

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The specified PDF file does not exist: {pdf_path}")
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:  # Check if text was extracted
                text += page_text + "\n"
    return text

if __name__ == "__main__":  # Corrected the main check
    pdf_path = "/home/rwik/code/satya/ModuleNotFound_Innov8_2.0/Final_Resumes/Resume_of_ID_0.pdf"
    text = extract_text_from_pdf(pdf_path)
    print(text)
