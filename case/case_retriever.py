from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import case_loader


def get_retriever(file_path: str):
    loader = case_loader.get_case_doc_loader(file_path)
    docs = loader.load()
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma.from_documents(documents=docs, embedding=embedding_model)
    return db.as_retriever()


if __name__ == "__main__":
    import json

    retriever = get_retriever("../data/case_gpt-4o.jsonl")
    query = "Stake ETH with Lido and deposit to Eigenpie"
    results = retriever.invoke(query)
    print(f"[0] page_content: {results[0].page_content}")
    print(f"[0] case_id: {results[0].metadata['case_id']}")
    steps = results[0].metadata["steps"].replace("'", '"')
    print(f"[0] steps: {json.loads(steps)}")
