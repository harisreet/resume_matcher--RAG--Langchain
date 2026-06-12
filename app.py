from src.resume_loader import load_resumes
from src.text_splitter import split_documents
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store
from src.retriever import get_retriever

documents = load_resumes("data/resumes")

chunks = split_documents(documents)

embedding_model = get_embedding_model()

vector_store = create_vector_store(
    chunks,
    embedding_model
)

retriever = get_retriever(vector_store)

query = "Python Developer"

results = retriever.invoke(query)

print("\nTop Matching Chunks:\n")

for i, doc in enumerate(results, start=1):
    print(f"Result {i}")
    print("Source:", doc.metadata.get("source"))
    print(doc.page_content[:300])
    print("-" * 50)