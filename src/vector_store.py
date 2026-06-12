from langchain_chroma import Chroma
from langchain_core.documents import Document


def create_vector_store(chunks, embedding_function):

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory="chroma_db"
    )

    return vector_store