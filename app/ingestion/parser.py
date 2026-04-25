from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

from app.config import Settings


class DocumentParser:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.splitter = SentenceSplitter(chunk_size=650, chunk_overlap=80)

    def parse_file(self, path: Path) -> list[Document]:
        if self.settings.use_llama_parse:
            return self._parse_with_llamaparse(path)
        reader = SimpleDirectoryReader(input_files=[str(path)])
        return reader.load_data()

    def parse_url(self, url: str) -> list[Document]:
        page_text = self._fetch_url_text(url)
        return [Document(text=page_text, metadata={"source": url})]

    def to_chunks(self, documents: list[Document]) -> list[str]:
        nodes = self.splitter.get_nodes_from_documents(documents)
        chunks = [node.get_content(metadata_mode="none").strip() for node in nodes]
        return [chunk for chunk in chunks if chunk]

    def _parse_with_llamaparse(self, path: Path) -> list[Document]:
        if not self.settings.llama_cloud_api_key:
            raise ValueError("LLAMA_CLOUD_API_KEY is required when USE_LLAMA_PARSE=true")
        from llama_parse import LlamaParse

        parser = LlamaParse(
            api_key=self.settings.llama_cloud_api_key,
            result_type="markdown",
        )
        return parser.load_data(str(path))

    @staticmethod
    def _fetch_url_text(url: str) -> str:
        response = httpx.get(url, follow_redirects=True, timeout=30.0)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)
