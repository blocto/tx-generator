import json
from langchain_community.document_loaders import JSONLoader


def _metadata_func(record: dict, metadata: dict) -> dict:
    """Function to extract metadata from the record"""
    metadata["case_id"] = record.get("case_id")
    metadata["total_steps"] = record.get("total_steps")
    metadata["steps"] = record.get("steps")
    return metadata


def get_case_doc_loader(file_path):
    return JSONLoader(
        file_path=file_path,
        jq_schema=".",
        content_key="description",
        json_lines=True,
        metadata_func=_metadata_func,
    )


if __name__ == "__main__":
    loader = get_case_doc_loader("../data/case_gpt-4o.jsonl")
    docs = loader.load()
    print(f"Total docs: {len(docs)}")
    print(f"Content: {docs[1].page_content[:100]}")
    print(f"Metadata: {docs[3].metadata}")

    # print(get_json_obj_from_jsonl("../data/case_gpt-4o.jsonl", 1))
