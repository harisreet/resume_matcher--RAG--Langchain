from src.resume_loader import load_resumes
from src.text_splitter import split_documents
from src.embeddings import get_embedding_model

documents = load_resumes("data/resumes")

chunks = split_documents(documents)

model = get_embedding_model()

vector = model.encode(chunks[0].page_content)

print("Vector Dimension:", len(vector))
print(vector[:10])