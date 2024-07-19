import base64
from dotenv import load_dotenv
import os
import requests
from tqdm import tqdm


class CaseDownloader:
    def __init__(self, branch: str, output_dir: str = "raw_data/"):
        load_dotenv()
        if branch not in ["develop", "main"]:
            raise ValueError("Branch not supported. Please use 'develop' or 'main'.")
        self.access_token: str = os.getenv("GITHUB_ACCESS_TOKEN")
        self.repo: str = "blocto/bento-interface"
        self.branch: str = branch
        self.github_api_url: str = "https://api.github.com"
        self.output_dir: str = output_dir

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.access_token}",
        }

    def _get_case_path(self) -> str:
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
        print(content_encoded)
        file_content = base64.b64decode(content_encoded)

        local_path = os.path.join(self.output_dir, path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        with open(local_path, "wb") as f:
            f.write(file_content)

    def download(self) -> None:
        """Download all bento cases from the repository."""
        case_paths = self._get_case_path()
        base_url = f"{self.github_api_url}/repos/{self.repo}/contents"

        for path in tqdm(case_paths, desc="Downloading cases", unit="file"):
            url = f"{base_url}/{path}"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            file_info = response.json()
            self._download_file(file_info, path)


if __name__ == "__main__":
    downloader = CaseDownloader(branch="develop")
    downloader.download()
