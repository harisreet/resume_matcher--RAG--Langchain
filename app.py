from src.resume_loader import load_resumes
from src.text_splitter import split_documents
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store
from src.retriever import get_retriever
from src.llm import get_llm
from src.rag_chain import build_rag_chain
from src.reranker import rerank   # 🔥 NEW

# 1. Load data
documents = load_resumes("data/resumes")

# 2. Split
chunks = split_documents(documents)

# 3. Embeddings
embedding_model = get_embedding_model()

# 4. Vector DB
vector_store = create_vector_store(chunks, embedding_model)

# 5. Retriever (MMR)
retriever = get_retriever(vector_store)

# 6. LLM
llm = get_llm()

# 7. Prompt
prompt = build_rag_chain()

# -----------------------
# QUERY
# -----------------------
job_description = """
We are looking for a Software Engineer with experience in:
- Linux
- C++
- Operating Systems
- Embedded Systems knowledge is a plus
"""
query = job_description
# 8. Retrieve (MMR)
docs = retriever.invoke(query)

# -----------------------
# 9. RERANK (🔥 NEW STEP)
# -----------------------
docs = rerank(query, docs)

# -----------------------
# 10. DEDUPLICATION
# -----------------------
unique_docs = {}

for doc in docs:
    source = doc.metadata.get("source", "unknown")
    if source not in unique_docs:
        unique_docs[source] = doc

docs = list(unique_docs.values())

# -----------------------
# DEBUG OUTPUT
# -----------------------
print("\n--- RERANKED RESUMES ---")

for i, doc in enumerate(docs):
    print(f"\n[{i+1}] {doc.metadata.get('source')}")
    print(doc.page_content[:250])

# -----------------------
# CONTEXT BUILDING
# -----------------------
context = "\n\n====================\n\n".join(
    f"""
Resume File: {doc.metadata.get('source', 'unknown')}
Content:
{doc.page_content}
""".strip()
    for doc in docs
)
# -----------------------
# LLM CALL
# -----------------------
messages = prompt.invoke({
    "context": context,
    "question": job_description
})

response = llm.invoke(messages)

print("\n--- FINAL ATS RANKING ---\n")
print(response.content)