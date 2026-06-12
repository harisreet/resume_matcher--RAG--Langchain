from src.resume_loader import load_resumes
from src.text_splitter import split_documents
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store

documents = load_resumes("data/resumes")

chunks = split_documents(documents)

embedding_model = get_embedding_model()

vector_store = create_vector_store(
    chunks,
    embedding_model
)

print("Vector Database Created Successfully")