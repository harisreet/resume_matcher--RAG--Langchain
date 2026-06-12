from src.resume_loader import load_resumes
from src.text_splitter import split_documents
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store
from src.retriever import get_retriever
from src.llm import get_llm
from src.rag_chain import build_rag_chain

documents = load_resumes("data/resumes")

chunks = split_documents(documents)

embedding_model = get_embedding_model()

vector_store = create_vector_store(
    chunks,
    embedding_model
)

retriever = get_retriever(vector_store)

llm = get_llm()

prompt = build_rag_chain()

question = "Who has Linux experience?"

docs = retriever.invoke(question)

context = "\n\n".join(
    doc.page_content for doc in docs
)

messages = prompt.invoke(
    {
        "context": context,
        "question": question
    }
)

response = llm.invoke(messages)

print(response.content)