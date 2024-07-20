import os
import glob
import asyncio
from typing import List, Iterator
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class CaseCodeLoader(BaseLoader):
    """
    Loads code snippets from the batch case codebase.
    """

    def _extract_case_name(self, file_path: str) -> tuple[str, str]:
        path_components = file_path.split(os.sep)
        case_index = path_components.index("cases")
        return path_components[case_index + 1], path_components[-1]

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
                    case_name, file_name = self._extract_case_name(file_path)
                    yield Document(
                        page_content=content,
                        metadata={"case": case_name, "file": file_name},
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
