from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from code_loader import CaseCodeLoader


def get_retriever():
    docs = CaseCodeLoader().load()
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma.from_documents(documents=docs, embedding=embedding_model)
    return vector_store.as_retriever()


if __name__ == "__main__":
    retriever = get_retriever()
    docs = retriever.invoke("withdraw usdc from compound")
    for doc in docs:
        print(doc.page_content[:200])
        print(doc.metadata)
        print("-------")
