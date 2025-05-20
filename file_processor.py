# Module for processing different file types 
import os
import docx
import fitz  # PyMuPDF

def extract_text_from_docx(file_path):
    """Extracts text from a .docx file."""
    try:
        doc = docx.Document(file_path)
        text = []
        for paragraph in doc.paragraphs:
            text.append(paragraph.text)
        return "\n".join(text)
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return None

def extract_text_from_pdf(file_path):
    """Extracts text from a .pdf file."""
    try:
        doc = fitz.open(file_path)
        text = []
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text.append(page.get_text())
        return "\n".join(text)
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return None

def process_file(file_path):
    """Extracts text from a supported file type."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == ".docx":
        return extract_text_from_docx(file_path)
    elif file_extension == ".pdf":
        return extract_text_from_pdf(file_path)
    elif file_extension == ".txt":
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading text file {file_path}: {e}")
            return None
    # Add handlers for other file types (e.g., .csv) here
    # elif file_extension == ".csv":
    #     pass # Implement CSV handling
    else:
        print(f"Unsupported file type: {file_extension}")
        return None

if __name__ == '__main__':
    # Example Usage (for testing)
    # Create dummy files for testing
    with open("test.txt", "w") as f:
        f.write("This is a test text file.")

    # You would need dummy .docx and .pdf files to test those functionalities
    # For now, we will just test the text file.

    text_content = process_file("test.txt")
    print(f"Text from test.txt: {text_content}")

    # Clean up dummy file
    os.remove("test.txt") 