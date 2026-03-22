"""
LangChain 文档爬虫
从 https://python.langchain.com/docs 爬取文档页面，保存为 Markdown

用法:
    python -m scripts.crawl_docs
    python -m scripts.crawl_docs --max-pages 50 --output ./data/langchain_docs
"""

import argparse
import re
import sys
import time
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()

# python.langchain.com/docs 已迁移，会重定向到 docs.langchain.com/oss/python/...
START_URL = "https://python.langchain.com/docs"
DOCS_DOMAIN = "docs.langchain.com"
DOCS_PATH_PREFIX = "/oss/python"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; LangChainDocsBot/1.0; educational use)"
}


def is_docs_url(url: str) -> bool:
    """只爬 docs.langchain.com/oss/python 下的页面，排除博客等"""
    parsed = urlparse(url)
    if parsed.netloc != DOCS_DOMAIN:
        return False
    path = parsed.path
    if not path.startswith(DOCS_PATH_PREFIX):
        return False
    # 排除 releases/changelog 等非概念文档页
    excluded = ["/oss/python/releases"]
    for exc in excluded:
        if path.startswith(exc):
            return False
    return True


def normalize_url(url: str, base: str) -> str | None:
    """转成绝对 URL，去掉 fragment 和 query"""
    full = urljoin(base, url)
    parsed = urlparse(full)
    return parsed._replace(query="", fragment="").geturl()


def extract_links(soup: BeautifulSoup, current_url: str) -> list[str]:
    """从页面中提取所有符合条件的文档链接"""
    links = []
    for tag in soup.find_all("a", href=True):
        href = tag["href"]
        if href.startswith("#") or href.startswith("mailto:"):
            continue
        norm = normalize_url(href, current_url)
        if norm and is_docs_url(norm):
            links.append(norm)
    return links


def extract_content(soup: BeautifulSoup) -> str | None:
    """
    提取页面正文并转为 Markdown。
    docs.langchain.com 的主内容在 div.grow.w-full 容器内。
    """
    # 主内容列：class 同时含 grow 和 w-full
    content_el = soup.find(
        "div",
        class_=lambda c: c and "grow" in c and "w-full" in c
    )
    if content_el is None:
        # 降级：找 prose 容器的最近大父级
        prose = soup.find(class_=lambda c: c and "prose" in (
            " ".join(c) if isinstance(c, list) else str(c)
        ))
        if prose:
            content_el = prose.parent
    if content_el is None:
        return None

    # 移除噪音：分页导航、侧边目录、编辑按钮、breadcrumb
    for noise in content_el.find_all(
        class_=lambda c: c and any(k in " ".join(c) for k in [
            "sidebar", "toc", "pagination", "breadcrumb",
            "edit", "feedback", "banner", "nav",
        ])
    ):
        noise.decompose()
    # 移除 script / style
    for tag in content_el.find_all(["script", "style", "button"]):
        tag.decompose()

    raw_md = md(
        str(content_el),
        heading_style="ATX",
        bullets="-",
        strip=["script", "style", "button", "nav", "footer"],
    )

    # 压缩超过 2 个的连续空行
    raw_md = re.sub(r"\n{3,}", "\n\n", raw_md)
    return raw_md.strip()


def url_to_filename(url: str) -> str:
    """把 URL path 转成安全文件名，如 /oss/python/langchain/lcel → langchain_lcel.md"""
    path = urlparse(url).path
    # 去掉 /oss/python/ 前缀
    path = re.sub(r"^/oss/python/?", "", path)
    name = path.strip("/").replace("/", "_")
    name = re.sub(r"[^\w\-]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return (name or "index") + ".md"


def crawl(start_url: str, output_dir: Path, max_pages: int, delay: float) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    visited: set[str] = set()
    queue: deque[str] = deque([start_url])
    saved = 0
    failed = 0

    session = requests.Session()
    session.headers.update(HEADERS)

    console.print(f"\n[bold blue]🕷  开始爬取 {start_url}[/bold blue]")
    console.print(f"   最多爬取: {max_pages} 页 | 间隔: {delay}s | 输出: {output_dir}\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("爬取进度", total=max_pages)

        while queue and saved < max_pages:
            url = queue.popleft()
            if url in visited:
                continue
            visited.add(url)

            short_path = urlparse(url).path
            progress.update(task, description=f"[cyan]{short_path[:55]:<55}")

            try:
                resp = session.get(url, timeout=15, allow_redirects=True)

                # 跟踪重定向后的真实 URL，避免重复
                final_url = resp.url
                parsed_final = urlparse(final_url)
                final_clean = parsed_final._replace(query="", fragment="").geturl()
                if final_clean != url and final_clean in visited:
                    continue
                visited.add(final_clean)

                if resp.status_code != 200:
                    console.print(f"  [yellow]⚠ {resp.status_code} {url}[/yellow]")
                    failed += 1
                    continue

                soup = BeautifulSoup(resp.text, "html.parser")

                # 发现新链接（基于最终 URL 解析）
                for link in extract_links(soup, final_clean):
                    if link not in visited:
                        queue.append(link)

                # 提取正文
                content = extract_content(soup)
                if not content or len(content) < 80:
                    console.print(f"  [dim]skip (no content): {short_path}[/dim]")
                    continue

                filename = url_to_filename(final_clean)
                filepath = output_dir / filename
                filepath.write_text(content, encoding="utf-8")
                saved += 1
                progress.advance(task)
                console.print(f"  [green]✓[/green] [{saved:3d}] {filename}")

            except requests.RequestException as e:
                console.print(f"  [red]✗ {url} — {e}[/red]")
                failed += 1

            time.sleep(delay)

    console.print(f"\n[bold green]✅ 完成！保存 {saved} 个文档，失败/跳过 {failed} 个[/bold green]")
    console.print(f"   输出目录: {output_dir.resolve()}\n")
    return saved


def main():
    parser = argparse.ArgumentParser(description="爬取 LangChain 文档")
    parser.add_argument("--start-url", default=START_URL, help="起始 URL")
    parser.add_argument("--max-pages", type=int, default=100, help="最多爬取页数")
    parser.add_argument("--output", default="./data/langchain_docs", help="输出目录")
    parser.add_argument("--delay", type=float, default=1.0, help="请求间隔秒数")
    args = parser.parse_args()

    saved = crawl(
        start_url=args.start_url,
        output_dir=Path(args.output),
        max_pages=args.max_pages,
        delay=args.delay,
    )
    sys.exit(0 if saved > 0 else 1)


if __name__ == "__main__":
    main()
