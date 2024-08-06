import os
import requests
import json
import base64
from dotenv import load_dotenv
from tqdm import tqdm


class CodeDownloader:
    def __init__(self, is_dev: bool = True, output_dir: str = "raw_data/"):
        load_dotenv()
        self.is_dev: bool = is_dev
        self.access_token: str = os.getenv("GITHUB_ACCESS_TOKEN")
        self.repo: str = "blocto/bento-interface"
        self.branch: str = "develop" if self.is_dev else "main"
        self.github_api_url: str = "https://api.github.com"
        self.output_dir: str = output_dir

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.access_token}",
        }

    def _get_case_path(self) -> list[str]:
        """Get all case paths from the repository."""
        base_url = (
            f"{self.github_api_url}/repos/{self.repo}/git/trees/"
            f"{self.branch}?recursive=1"
        )
        response = requests.get(base_url, headers=self.headers)
        response.raise_for_status()
        all_files = response.json()["tree"]
        return [
            file["path"]
            for file in all_files
            if file["path"].startswith("cases/")
            and file["path"].endswith(".ts")
            and len(file["path"].split("/")) > 2
        ]

    def _download_file(self, file_info: dict, path: str) -> None:
        """Download a single file from the repository."""
        content_encoded = file_info["content"]
        file_content = base64.b64decode(content_encoded)

        with open(os.path.join(self.output_dir, path), "wb") as f:
            f.write(file_content)

    def _download_meta(self):
        """Download the metadata file from the repository."""
        import json

        env = "dev" if self.is_dev else "release"
        url = f"https://bento-batch-{env}.netlify.app/case/api/meta"
        response = requests.get(url)
        response.raise_for_status()
        cases = response.json()["cases"]
        # Sort the cases by id alphabetically
        cases_sorted = sorted(cases, key=lambda case: case["id"])
        # Write the metadata to a file
        with open(os.path.join(self.output_dir, "meta.json"), "w") as f:
            json.dump(cases_sorted, f)

    def download(self) -> int:
        """Download all bento cases from the repository."""
        case_paths = self._get_case_path()
        base_url = f"{self.github_api_url}/repos/{self.repo}/contents"

        # Check if output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Download metadata
        self._download_meta()

        # Download all cases
        for path in tqdm(case_paths, desc="Downloading files", unit="file"):
            local_dir = os.path.dirname(os.path.join(self.output_dir, path))
            os.makedirs(local_dir, exist_ok=True)
            url = f"{base_url}/{path}"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            file_info = response.json()
            self._download_file(file_info, path)
        return len(case_paths)


def get_metadata():
    with open("raw_data/meta.json", "r") as f:
        cases = json.load(f)

    metadata = {
        case["id"]: {
            "chain_id": case["chain_id"],
            "preview_txn_count": case["preview_txn_count"],
        }
        for case in cases
    }
    return metadata


if __name__ == "__main__":
    downloader = CodeDownloader(is_dev=True)
    downloader.download()

    print(f"Case amount in metadata: {len(get_metadata())}")
