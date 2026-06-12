from src.resume_loader import load_resumes
from src.text_splitter import split_documents
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store
from src.retriever import get_retriever
from src.llm import get_llm
from src.rag_chain import build_rag_chain

# 1. Load data
documents = load_resumes("data/resumes")

# 2. Split into chunks
chunks = split_documents(documents)

# 3. Embeddings
embedding_model = get_embedding_model()

# 4. Vector DB
vector_store = create_vector_store(chunks, embedding_model)

# 5. Retriever
retriever = get_retriever(vector_store)

# 6. LLM
llm = get_llm()

# 7. Prompt ONLY
prompt = build_rag_chain()

# 8. Query
question = "Who has Linux experience?"

# 9. Retrieve docs
docs = retriever.invoke(question)

# 10. Build context (clean version)
context = "\n\n".join(
    f"Resume: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
    for doc in docs
)

# 11. Prompt → LLM
messages = prompt.invoke({
    "context": context,
    "question": question
})

response = llm.invoke(messages)

print(response.content)