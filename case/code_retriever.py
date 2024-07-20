from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from case.code_loader import CaseCodeLoader


def get_retriever():
    docs = CaseCodeLoader().load()
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma.from_documents(documents=docs, embedding=embedding_model)
    return vector_store.as_retriever()
