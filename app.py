from src.pdf_loader import load_resume

pdf_path = "data/resumes/60004873.pdf"

documents = load_resume(pdf_path)

print(f"Total Pages: {len(documents)}\n")

for i, doc in enumerate(documents, start=1):
    print(f"----- Page {i} -----")
    print(doc.page_content)
    print()