from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import BaseNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


Settings.llm = None


_HEADER_SPLIT_RE = re.compile(r"[\\/]+")
MIN_CONTENT_CHARS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest parsed California DMV handbook markdown into Qdrant "
            "and export a stable corpus.jsonl for retrieval/eval construction."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input parsed markdown file or directory.",
    )
    parser.add_argument(
        "--collection",
        default="dmv_handbook_v5",
        help="Qdrant collection name.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=450,
        help=(
            "SentenceSplitter chunk size. Recommended for this DMV handbook: "
            "450 tokens-ish units."
        ),
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=80,
        help=(
            "SentenceSplitter chunk overlap. Recommended for this DMV handbook: "
            "80 tokens-ish units."
        ),
    )
    parser.add_argument(
        "--embedding-model",
        default=(
            "Alibaba-NLP/"
            "gte-modernbert-base"
        ),
        help="Embedding model name.",
    )
    parser.add_argument(
        "--qdrant-host",
        default="localhost",
        help="Qdrant host.",
    )
    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=6333,
        help="Qdrant port.",
    )
    parser.add_argument(
        "--corpus-version",
        default="dmv_ca_2025_v4_chunk450_overlap80",
        help=(
            "Stable corpus version string. Change this whenever chunking, "
            "source parsing, or corpus content changes."
        ),
    )
    parser.add_argument(
        "--corpus-out",
        default="evaluation/datasets/raw_extracted_data/corpus.jsonl",
        help="Where to export corpus.jsonl.",
    )
    parser.add_argument(
        "--report-out",
        default="RAG/ingesting-vdb/ingestion_report.json",
        help="Where to write ingestion report.",
    )
    parser.add_argument(
        "--recreate-collection",
        action="store_true",
        help=(
            "Delete the Qdrant collection before ingesting. Use this for clean "
            "re-ingestion after changing chunk size/overlap/corpus version."
        ),
    )

    return parser.parse_args()


def _normalize_for_hash(text: str) -> str:
    return " ".join((text or "").split())


def _sha1_short(text: str, length: int = 24) -> str:
    normalized = _normalize_for_hash(text)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:length]


def assign_stable_chunk_ids(
    nodes: List[BaseNode],
    *,
    corpus_version: str,
) -> List[BaseNode]:
    counters: dict[str, int] = {}

    for node in nodes:
        metadata = node.metadata or {}

        file_name = str(metadata.get("file_name") or "unknown_file")
        heading_path = str(metadata.get("heading_path") or "unknown_heading")
        section_id = str(metadata.get("section_id") or "unknown_section")

        group_key = f"{file_name}|{section_id}|{heading_path}"
        chunk_index = counters.get(group_key, 0)
        counters[group_key] = chunk_index + 1

        text = node.get_content() or ""
        text_hash = _sha1_short(text, length=16)

        stable_chunk_id = _sha1_short(
            f"{corpus_version}|{group_key}|{chunk_index}|{text_hash}",
            length=24,
        )

        metadata["stable_chunk_id"] = stable_chunk_id
        metadata["chunk_index"] = chunk_index
        metadata["text_hash"] = text_hash
        metadata["corpus_version"] = corpus_version

        node.metadata = metadata

    return nodes


def load_documents(input_path: str):
    path = Path(input_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if path.is_file():
        return SimpleDirectoryReader(input_files=[str(path)]).load_data()

    return SimpleDirectoryReader(
        input_dir=str(path),
        recursive=False,
    ).load_data()


def parse_markdown_nodes(documents) -> List[BaseNode]:
    markdown_parser = MarkdownNodeParser(
        include_metadata=True,
        include_prev_next_rel=True,
    )

    return markdown_parser.get_nodes_from_documents(documents)


def _infer_heading_level_and_title(text: str) -> Tuple[int, str, str]:
    """
    Return (level, title, body_text).

    - level: number of leading '#' in the first line if it is a heading; else 0
    - title: heading text after '#'
    - body_text: remaining text after the heading line
    """
    if not text:
        return 0, "", ""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    first = lines[0].strip()

    if first.startswith("#"):
        level = 0

        for char in first:
            if char == "#":
                level += 1
            else:
                break

        title = first[level:].strip(" .")
        body = "\n".join(lines[1:]).strip()

        return level, title, body

    return 0, "", normalized.strip()


def _is_empty_parent_heading(node: BaseNode) -> bool:
    """
    Return True if the node is an H1/H2 heading with no real body content.
    """
    text = (node.get_content() or "").strip()
    level, _, body = _infer_heading_level_and_title(text)

    if level in (1, 2):
        return not body or len(body.strip()) < MIN_CONTENT_CHARS

    return False


def _normalize_header_path(
    raw: Optional[str],
    *,
    text: Optional[str] = None,
) -> List[str]:
    """
    Normalize markdown header.

    Examples:
    - "/A\\r/B\\r/" -> ["A", "B"]
    - "/A/A/B/" -> ["A", "B"]
    """
    if raw:
        cleaned = raw.replace("\r", "").strip()
        parts = [
            part.strip()
            for part in _HEADER_SPLIT_RE.split(cleaned)
            if part.strip()
        ]

        normalized: List[str] = []

        for part in parts:
            if not normalized or normalized[-1] != part:
                normalized.append(part)

        if normalized:
            return normalized

    if text:
        first_line = (
            text.replace("\r\n", "\n")
            .replace("\r", "\n")
            .split("\n", 1)[0]
            .strip()
        )

        match = re.match(r"^\s{0,3}(#+)\s*(.+?)\s*$", first_line)

        if match:
            title = re.sub(r"\s+", " ", match.group(2)).strip(" .#")
            return [title] if title else []

    return []


def _detect_section_id(path_titles: List[str], file_name: str) -> str:
    """
    Prefer SECTION from heading path; fallback to file name.
    """
    for title in path_titles:
        match = re.search(r"section\s+(\d+)", title, flags=re.I)

        if match:
            return f"SECTION {int(match.group(1))}"

    base_name = Path(file_name or "").name.lower()
    match = re.search(r"sec(?:tion)?[_\-\s]?(\d+)", base_name)

    if match:
        return f"SECTION {int(match.group(1))}"

    return "SECTION ?"


def enrich_md_nodes(raw_nodes: List[BaseNode]) -> List[BaseNode]:
    enriched_nodes: List[BaseNode] = []

    for node in raw_nodes:
        metadata = node.metadata or {}
        file_name = metadata.get("file_name", "")

        text = node.get_content() or ""
        path_titles = _normalize_header_path(
            metadata.get("header_path"),
            text=text,
        )

        level, current_title, _ = _infer_heading_level_and_title(text)

        full_titles = list(path_titles)

        if (
            level > 0
            and current_title
            and (not full_titles or full_titles[-1] != current_title)
        ):
            full_titles.append(current_title)

        heading_path = " > ".join(full_titles) if full_titles else ""
        ancestors = [
            {"level": index + 1, "title": title}
            for index, title in enumerate(path_titles)
        ]

        depth = level if level > 0 else len(full_titles) if full_titles else 0
        parent_near = path_titles[-1] if path_titles else None
        parent_major = path_titles[1] if len(path_titles) >= 2 else None
        parent_topics = list(path_titles)
        section_id = metadata.get("section_id") or _detect_section_id(
            path_titles,
            file_name,
        )

        metadata["heading_path"] = heading_path
        metadata["ancestors"] = ancestors
        metadata["depth"] = depth
        metadata["parent_topics"] = parent_topics
        metadata["section_id"] = section_id
        metadata["file_name"] = file_name

        if parent_near:
            metadata["parent_near"] = parent_near

        if parent_major:
            metadata["parent_major"] = parent_major

        node.metadata = metadata

        if _is_empty_parent_heading(node):
            continue

        enriched_nodes.append(node)

    return enriched_nodes


def recreate_qdrant_collection_if_requested(
    *,
    client: QdrantClient,
    collection: str,
    recreate_collection: bool,
) -> None:
    if not recreate_collection:
        return

    try:
        if client.collection_exists(collection):
            print(f"Deleting existing Qdrant collection: {collection}")
            client.delete_collection(collection_name=collection)
    except Exception as exc:
        print(f"collection_exists check failed, trying delete_collection directly: {exc}")
        try:
            client.delete_collection(collection_name=collection)
            print(f"Deleted existing Qdrant collection: {collection}")
        except Exception as delete_exc:
            print(f"delete_collection skipped or failed: {delete_exc}")


def build_index_from_nodes(
    nodes: List[BaseNode],
    *,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
    qdrant_host: str,
    qdrant_port: int,
    collection: str,
    corpus_version: str,
    recreate_collection: bool,
) -> Tuple[VectorStoreIndex, List[BaseNode]]:
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=embedding_model,
        normalize=True,
    )

    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        paragraph_separator="\n\n",
    )

    child_nodes = splitter.get_nodes_from_documents(nodes)
    child_nodes = assign_stable_chunk_ids(
        child_nodes,
        corpus_version=corpus_version,
    )

    client = QdrantClient(
        host=qdrant_host,
        port=qdrant_port,
    )

    recreate_qdrant_collection_if_requested(
        client=client,
        collection=collection,
        recreate_collection=recreate_collection,
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection,
        enable_hybrid=True,
        fastembed_sparse_model="Qdrant/bm25",
        batch_size=20,
    )

    storage = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        [],
        storage_context=storage,
        show_progress=True,
    )

    index.insert_nodes(child_nodes)

    return index, child_nodes


def export_corpus_jsonl(
    nodes: List[BaseNode],
    output_path: str,
) -> None:
    """
    Export chunked corpus for retrieval benchmark construction.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for node in nodes:
            metadata = node.metadata or {}

            record = {
                "stable_chunk_id": metadata.get("stable_chunk_id"),
                "text": node.get_content() or "",
                "section_id": metadata.get("section_id"),
                "heading_path": metadata.get("heading_path"),
                "file_name": metadata.get("file_name"),
                "chunk_index": metadata.get("chunk_index"),
                "text_hash": metadata.get("text_hash"),
                "corpus_version": metadata.get("corpus_version"),
                "parent_topics": metadata.get("parent_topics", []),
                "parent_near": metadata.get("parent_near"),
                "parent_major": metadata.get("parent_major"),
            }

            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_ingestion_report(
    *,
    args: argparse.Namespace,
    documents_count: int,
    raw_node_count: int,
    enriched_node_count: int,
    node_count: int,
) -> None:
    report = {
        "input": str(args.input),
        "collection": args.collection,
        "embedding_model": args.embedding_model,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "qdrant_host": args.qdrant_host,
        "qdrant_port": args.qdrant_port,
        "corpus_version": args.corpus_version,
        "corpus_out": str(args.corpus_out),
        "document_count": documents_count,
        "raw_markdown_node_count": raw_node_count,
        "enriched_markdown_node_count": enriched_node_count,
        "node_count": node_count,
        "recreate_collection": bool(args.recreate_collection),
    }

    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))


def main() -> None:
    args = parse_args()

    documents = load_documents(args.input)
    markdown_nodes = parse_markdown_nodes(documents)
    preprocessed_nodes = enrich_md_nodes(markdown_nodes)

    _, child_nodes = build_index_from_nodes(
        preprocessed_nodes,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        collection=args.collection,
        corpus_version=args.corpus_version,
        recreate_collection=args.recreate_collection,
    )

    export_corpus_jsonl(
        child_nodes,
        args.corpus_out,
    )

    write_ingestion_report(
        args=args,
        documents_count=len(documents),
        raw_node_count=len(markdown_nodes),
        enriched_node_count=len(preprocessed_nodes),
        node_count=len(child_nodes),
    )


if __name__ == "__main__":
    main()