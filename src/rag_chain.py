from langchain_core.prompts import ChatPromptTemplate

def build_rag_chain():

    prompt = ChatPromptTemplate.from_template("""
You are a resume screening assistant.

Answer ONLY using the provided context.

If the resume filename is available in the metadata, mention it.

Context:
{context}

Question:
{question}
""")

    return prompt