import os
import glob
import asyncio
from typing import List, Iterator
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class CaseCodeLoader(BaseLoader):
    def _extract_protocol_name(self, file_path: str) -> str:
        path_components = file_path.split(os.sep)
        case_index = path_components.index("cases")
        return path_components[case_index + 1]

    def load(self) -> List[Document]:
        """Load data into Document objects."""
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        directory_path = "raw_data/cases"
        file_pattern = os.path.join(directory_path, "**", "*")
        for file_path in glob.iglob(file_pattern, recursive=True):
            if os.path.isfile(file_path):
                with open(file_path, "r") as file:
                    content = file.read()
                    yield Document(
                        page_content=content,
                        metadata={"protocol": self._extract_protocol_name(file_path)},
                    )


async def async_case_loader():
    loader = CaseCodeLoader()
    async for doc in loader.alazy_load():
        print()
        print(type(doc))
        print(doc.metadata)


if __name__ == "__main__":
    loader = CaseCodeLoader()

    asyncio.run(async_case_loader())
