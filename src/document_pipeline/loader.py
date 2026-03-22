"""
文档加载器 - 支持多种格式的文档读取
支持: Markdown (.md), HTML (.html), 纯文本 (.txt)

面试要点：为什么不用PDF？因为LangChain文档是网页/Markdown格式，
选择合适的loader体现你对数据源的理解。
"""

from pathlib import Path
from dataclasses import dataclass, field
from rich.console import Console

console = Console()


@dataclass
class Document:
    """统一的文档数据结构"""
    content: str                        # 文档正文
    metadata: dict = field(default_factory=dict)  # 元数据：来源、文件名、格式等

    def __repr__(self) -> str:
        preview = self.content[:80] + "..." if len(self.content) > 80 else self.content
        return f"Document(source={self.metadata.get('source', '?')}, len={len(self.content)}, preview='{preview}')"


class DocumentLoader:
    """
    从本地文件夹加载文档
    
    用法:
        loader = DocumentLoader("./data/langchain_docs")
        docs = loader.load_all()
    """

    SUPPORTED_EXTENSIONS = {".md", ".html", ".txt", ".mdx"}

    def __init__(self, docs_dir: str):
        self.docs_dir = Path(docs_dir)
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"文档目录不存在: {self.docs_dir}")

    def load_single(self, file_path: Path) -> Document | None:
        """加载单个文件，返回 Document 对象"""
        try:
            text = file_path.read_text(encoding="utf-8")

            # HTML 文件：提取纯文本
            if file_path.suffix == ".html":
                text = self._strip_html(text)

            # 跳过空文件
            if len(text.strip()) < 50:
                return None

            return Document(
                content=text.strip(),
                metadata={
                    "source": str(file_path.relative_to(self.docs_dir)),
                    "filename": file_path.name,
                    "format": file_path.suffix.lstrip("."),
                    "char_count": len(text),
                },
            )
        except Exception as e:
            console.print(f"[yellow]⚠ 跳过文件 {file_path.name}: {e}[/yellow]")
            return None

    def load_all(self) -> list[Document]:
        """递归加载目录下所有支持的文件"""
        docs: list[Document] = []
        files = sorted(self.docs_dir.rglob("*"))

        for f in files:
            if f.suffix in self.SUPPORTED_EXTENSIONS and f.is_file():
                doc = self.load_single(f)
                if doc:
                    docs.append(doc)

        console.print(f"[green]✓ 成功加载 {len(docs)} 篇文档[/green]")
        return docs

    @staticmethod
    def _strip_html(html: str) -> str:
        """从HTML中提取纯文本"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            # 移除 script 和 style 标签
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            return soup.get_text(separator="\n", strip=True)
        except ImportError:
            # 如果没装 bs4，用简单的正则
            import re
            return re.sub(r"<[^>]+>", "", html)
