from langchain_core.prompts import ChatPromptTemplate

def build_rag_chain():
    return ChatPromptTemplate.from_template("""
You are a strict resume screening assistant.

RULES:
- Answer ONLY from provided context
- If not found, say "Not found in resumes"
- Mention resume filename if available

Context:
{context}

Question:
{question}
""")