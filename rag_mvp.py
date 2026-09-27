"""Local hybrid Markdown retriever for the VERONICA knowledge base."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import unicodedata
import platform
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised on an unconfigured machine
    raise SystemExit("Falta NumPy. Instala dependencias con: pip install -r requirements-rag.txt") from exc


ROOT = Path(__file__).resolve().parent
INDEX_PATH = ROOT / ".veronica" / "rag_index.json"
QUERY_PATH = ROOT / "rag_reference_queries.json"
BASELINE_PATH = ROOT / "rag_reference_baseline.json"
MAX_CHUNK_CHARS = 1400
MAX_VOCAB = 6000
DEFAULT_RANKING_K = [1, 3, 5, 10]
STOPWORDS = set("""a al algo algunas algunos ante antes como con contra cual cuando de del desde donde dos el ella ellas ellos en entre era es esa esas ese eso esos esta estaba estas este esto estos fue ha hacia hasta hay la las le les lo los mas me mi mientras muy ni no nos o otra otras otro otros para pero por porque que se sin sobre su sus te tiene todo tu un una unas uno unos y ya the a an and are as at be been by for from has have if in into is it of on or our that the their then there these this to was we were what when which with will with""".split())


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in text if not unicodedata.combining(char))


_SPANISH_SUFFIXES = ("aciones", "acion", "ar", "er", "ir")


def _spanish_lexical_key(token: str) -> str:
    """Reduce a few productive Spanish verb/action-noun endings for retrieval."""
    for suffix in _SPANISH_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[:-len(suffix)]
    return token


def tokenize(text: str) -> list[str]:
    return [_spanish_lexical_key(token)
            for token in re.findall(r"[a-z0-9]+(?:[._+#-][a-z0-9+#-]+)*", _normalize(text))
            if len(token) > 1 and token not in STOPWORDS]


def _frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end < 0:
        return {}, text
    metadata: dict[str, str] = {}
    for line in text[4:end].splitlines():
        match = re.match(r"^([A-Za-z_][\w-]*):\s*(.*?)\s*$", line)
        if match and not match.group(2).startswith("-"):
            value = match.group(2).strip().strip('"\'')
            if value and value.lower() not in {"unknown", "null", "none"}:
                metadata[match.group(1)] = value
    return metadata, text[end + 4:].lstrip("\r\n")


def _slug(text: str) -> str:
    slug = _normalize(text).lower()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    return re.sub(r"[\s-]+", "-", slug).strip("-")


def _markdown_blocks(text: str) -> list[tuple[str | None, str]]:
    blocks: list[tuple[str | None, str]] = []
    headings: list[str | None] = [None] * 6
    content: list[str] = []
    fenced = False

    def flush() -> None:
        value = "\n".join(content).strip()
        if value:
            blocks.append(("\n".join(h for h in headings if h), value))
        content.clear()

    for line in text.splitlines():
        if re.match(r"^\s*(```|~~~)", line):
            fenced = not fenced
        heading = None if fenced else re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if heading:
            flush()
            level, title = len(heading.group(1)), heading.group(2).strip()
            headings[level - 1] = title
            for i in range(level, 6):
                headings[i] = None
            continue
        if not line.strip() and not fenced:
            flush()
        else:
            content.append(line)
    flush()
    return blocks


def _split_long(block: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    if len(block) <= limit:
        return [block]
    pieces: list[str] = []
    current = ""
    # Keep lines intact where possible (tables and fenced code are usually line-based).
    for line in block.splitlines(keepends=True):
        if len(line) > limit:
            if current:
                pieces.append(current.strip())
                current = ""
            pieces.extend(line[i:i + limit].strip() for i in range(0, len(line), limit))
        elif current and len(current) + len(line) > limit:
            pieces.append(current.strip())
            current = line
        else:
            current += line
    if current.strip():
        pieces.append(current.strip())
    return [piece for piece in pieces if piece]


def load_documents() -> list[dict[str, Any]]:
    documents = []
    for folder, doc_type in (("knowledge", "knowledge"), ("playbook", "playbook"), ("playbooks", "playbook")):
        base = ROOT / folder
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.md")):
            raw = path.read_text(encoding="utf-8-sig")
            metadata, body = _frontmatter(raw)
            headings = [title for title, _ in _markdown_blocks(body) if title]
            h1 = next((h.split("\n", 1)[0] for h in headings if "\n" not in h), None)
            title = metadata.get("title") or h1 or path.stem.replace("_", " ")
            relative = path.relative_to(ROOT).as_posix()
            topic = metadata.get("knowledge_area") or metadata.get("topic") or metadata.get("subject")
            grouped: list[tuple[str | None, str]] = []
            pending_path: str | None = None
            pending: list[str] = []

            def flush_pending() -> None:
                nonlocal pending
                if pending:
                    grouped.append((pending_path, "\n\n".join(pending)))
                    pending = []

            for section_path, block in _markdown_blocks(body):
                for piece in _split_long(block):
                    if pending and (pending_path != section_path or len("\n\n".join(pending)) + len(piece) + 2 > MAX_CHUNK_CHARS):
                        flush_pending()
                    pending_path = section_path
                    pending.append(piece)
            flush_pending()
            chunks = []
            for number, (section_path, content) in enumerate(grouped, start=1):
                section = section_path.split("\n")[-1] if section_path else title
                header = f"Document: {title}\nSection: {section_path or title}\n"
                text = header + content
                digest = hashlib.sha256((relative + "\0" + (section_path or "") + "\0" + content).encode("utf-8")).hexdigest()[:20]
                chunks.append({
                    "chunk_id": digest,
                    "chunk_number": number,
                    "source_path": relative,
                    "document_type": doc_type,
                    "document_title": title,
                    "section_title": section,
                    "section_path": section_path or title,
                    "topic": topic,
                    "text": text,
                    "anchor": _slug(section) if section_path else "",
                    "tokens": tokenize(text),
                })
            if chunks:
                documents.append({"path": relative, "chunks": chunks})
    return documents


def source_manifest() -> dict[str, str]:
    manifest: dict[str, str] = {}
    for folder in ("knowledge", "playbook", "playbooks"):
        base = ROOT / folder
        if base.is_dir():
            for path in sorted(base.rglob("*.md")):
                manifest[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return manifest


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _effective_configuration(index: dict[str, Any], query_file: Path, *,
                             per_section_limit: int | None,
                             deduplicate_exact: bool,
                             ranking_k: list[int]) -> dict[str, Any]:
    manifest = index["source_manifest"]
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":")).encode("utf-8")
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "chunking": {"max_chunk_chars": MAX_CHUNK_CHARS,
                     "merge_adjacent_blocks_until_limit": True,
                     "scope": ["knowledge", "playbook", "playbooks"]},
        "tokenization": {"normalization": "Unicode NFKD + casefold; remove combining marks; conservative Spanish suffix reduction",
                         "pattern": r"[a-z0-9]+(?:[._+#-][a-z0-9+#-]+)*",
                         "spanish_suffixes": list(_SPANISH_SUFFIXES),
                         "minimum_token_length": 2,
                         "stopwords_count": len(STOPWORDS),
                         "stopwords_sha256": _sha256_bytes(" ".join(sorted(STOPWORDS)).encode("utf-8"))},
        "vocabulary": {"max_vocab": MAX_VOCAB, "term_order": "descending_total_frequency_then_lexical",
                       "idf": "log(1 + (N - df + 0.5) / (df + 0.5))",
                       "tf_idf": "(1 + log(tf)) * idf; L2 normalized"},
        "ranking": {"lexical": "BM25; normalized by maximum BM25 among allowed results",
                    "bm25_k1": 1.5, "bm25_b": 0.75,
                    "semantic": "LSA projection of normalized TF-IDF; positive cosine",
                    "max_lsa_components": 48,
                    "effective_lsa_components": len(index["components"]),
                    "lexical_weight": 0.55, "semantic_weight": 0.45,
                    "tie_break": ["source_path", "chunk_number"]},
        "ranking_k": ranking_k,
        "filters": {"supported": ["document_type", "topic"],
                    "per_case": {case["id"]: {key: case[key] for key in ("document_type", "topic") if case.get(key)}
                                 for case in _load_query_cases(query_file)[0]}},
        "diversity": {"per_section_limit": per_section_limit,
                      "deduplicate_exact": deduplicate_exact,
                      "section_key": ["source_path", "section_path"]},
        "queries_sha256": _sha256_bytes(query_file.read_bytes()),
        "corpus_manifest_sha256": _sha256_bytes(manifest_bytes),
        "corpus_manifest_files": len(manifest),
        "indexed_documents": index["documents"],
        "corpus_files": manifest,
        "relevance_definition": "Positive hit iff one returned chunk has exact expected_path, expected_section as substring of section_path, and all expected_terms in the same chunk (case-insensitive). Negative cases expect zero results and are excluded from ranking metrics.",
    }


def _load_query_cases(query_file: Path) -> tuple[list[dict[str, Any]], list[int]]:
    reference_data = json.loads(query_file.read_text(encoding="utf-8"))
    if isinstance(reference_data, list):
        return reference_data, DEFAULT_RANKING_K
    cases = reference_data["cases"]
    ranking_k = sorted(set(int(k) for k in reference_data.get("ranking_k", DEFAULT_RANKING_K)))
    return cases, ranking_k


def _build_vectors(chunks: list[dict[str, Any]]) -> tuple[list[str], list[float], list[list[float]], list[list[float]]]:
    df: Counter[str] = Counter()
    tf: Counter[str] = Counter()
    for chunk in chunks:
        counts = Counter(chunk["tokens"])
        df.update(counts.keys())
        tf.update(counts)
    vocab = [term for term, _ in sorted(tf.items(), key=lambda pair: (-pair[1], pair[0]))[:MAX_VOCAB]]
    positions = {term: i for i, term in enumerate(vocab)}
    n_docs = len(chunks)
    idf = [math.log(1 + (n_docs - df[term] + 0.5) / (df[term] + 0.5)) for term in vocab]
    matrix = np.zeros((n_docs, len(vocab)), dtype=np.float32)
    for row, chunk in enumerate(chunks):
        counts = Counter(chunk["tokens"])
        weighted = []
        for term, count in counts.items():
            if term in positions:
                col = positions[term]
                value = (1 + math.log(count)) * idf[col]
                matrix[row, col] = value
                weighted.append(value * value)
        norm = math.sqrt(sum(weighted))
        if norm:
            matrix[row] /= norm
    components: np.ndarray
    semantic: np.ndarray
    if n_docs >= 2 and len(vocab) >= 2:
        _, singular, vt = np.linalg.svd(matrix, full_matrices=False)
        rank = max(1, min(48, n_docs - 1, len(vocab)))
        components = vt[:rank]
        semantic = matrix @ components.T
        # Ignore numerical noise in empty/degenerate corpora.
        if singular.size and singular[0] > 1e-12:
            semantic = matrix @ components.T
        components = components.astype(np.float32)
        semantic = semantic.astype(np.float32)
    else:
        components = np.zeros((0, len(vocab)), dtype=np.float32)
        semantic = np.zeros((n_docs, 0), dtype=np.float32)
    return vocab, idf, components.tolist(), semantic.tolist()


def build_index(path: Path = INDEX_PATH) -> dict[str, Any]:
    documents = load_documents()
    chunks = [chunk for document in documents for chunk in document["chunks"]]
    vocab, idf, components, semantic = _build_vectors(chunks)
    index = {
        "format_version": 1,
        "documents": len(documents),
        "chunk_count": len(chunks),
        "source_manifest": source_manifest(),
        "vocabulary": vocab,
        "idf": idf,
        "components": components,
        "semantic_vectors": semantic,
        "chunks": [{k: v for k, v in chunk.items() if k != "tokens"} | {"tokens": chunk["tokens"]} for chunk in chunks],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return index


def _cosine_scores(query: list[float], vectors: np.ndarray) -> np.ndarray:
    qnorm = float(np.linalg.norm(query))
    norms = np.linalg.norm(vectors, axis=1)
    if qnorm == 0:
        return np.zeros(len(vectors), dtype=np.float32)
    return (vectors @ np.asarray(query, dtype=np.float32)) / np.maximum(norms * qnorm, 1e-12)


def _select_results(candidates: list[dict[str, Any]], top_k: int, *,
                    per_section_limit: int | None = None,
                    deduplicate_exact: bool = False) -> list[dict[str, Any]]:
    """Filter an already-ranked result list without changing retained order."""
    if per_section_limit is not None and per_section_limit < 1:
        raise ValueError("per_section_limit debe ser un entero positivo o None")
    selected = []
    section_counts: Counter[tuple[str, str]] = Counter()
    seen_texts: set[str] = set()
    for result in candidates:
        if deduplicate_exact and result["text"] in seen_texts:
            continue
        section_key = (result["source_path"], result["section_path"])
        if per_section_limit is not None and section_counts[section_key] >= per_section_limit:
            continue
        selected.append(result)
        section_counts[section_key] += 1
        seen_texts.add(result["text"])
        if len(selected) >= top_k:
            break
    return selected


def search(query: str, *, top_k: int = 5, document_type: str | None = None,
           topic: str | None = None, index_path: Path = INDEX_PATH,
           per_section_limit: int | None = 2,
           deduplicate_exact: bool = False) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []
    if not index_path.exists() or json.loads(index_path.read_text(encoding="utf-8")).get("source_manifest") != source_manifest():
        build_index(index_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    chunks = index["chunks"]
    allowed = [i for i, chunk in enumerate(chunks)
               if (document_type is None or chunk["document_type"] == document_type)
               and (topic is None or (chunk.get("topic") or "").casefold() == topic.casefold())]
    if not allowed:
        return []
    vocab = index["vocabulary"]
    positions = {term: i for i, term in enumerate(vocab)}
    idf = index["idf"]
    query_tokens = tokenize(query)
    if not any(term in positions for term in query_tokens):
        return []
    query_counts = Counter(query_tokens)
    qvector = np.zeros(len(vocab), dtype=np.float32)
    for term, count in query_counts.items():
        if term in positions:
            col = positions[term]
            qvector[col] = (1 + math.log(count)) * idf[col]
    norm = float(np.linalg.norm(qvector))
    if norm:
        qvector /= norm
    doc_freq: Counter[str] = Counter()
    for chunk in chunks:
        doc_freq.update(set(chunk["tokens"]))
    n = len(chunks)
    avgdl = sum(len(chunk["tokens"]) for chunk in chunks) / max(n, 1)
    bm25 = np.zeros(len(chunks), dtype=np.float32)
    k1, b = 1.5, 0.75
    for i, chunk in enumerate(chunks):
        counts = Counter(chunk["tokens"])
        length = len(chunk["tokens"])
        for term, qcount in query_counts.items():
            if term not in positions or term not in counts:
                continue
            df = doc_freq[term]
            term_idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            freq = counts[term]
            bm25[i] += term_idf * freq * (k1 + 1) / (freq + k1 * (1 - b + b * length / max(avgdl, 1)))
    bm25_scale = float(np.max(bm25[allowed])) if allowed else 0.0
    lexical = bm25 / bm25_scale if bm25_scale else bm25
    components = np.asarray(index["components"], dtype=np.float32)
    semantic_vectors = np.asarray(index["semantic_vectors"], dtype=np.float32)
    if components.size and qvector.size:
        query_semantic = qvector @ components.T
        semantic = np.maximum(0.0, _cosine_scores(query_semantic, semantic_vectors))
    else:
        semantic = np.zeros(n, dtype=np.float32)
    # Equal blend; scores are ranking signals, not confidence probabilities.
    combined = 0.55 * lexical + 0.45 * semantic
    candidates = []
    for i in sorted(allowed, key=lambda j: (-float(combined[j]), chunks[j]["source_path"], chunks[j]["chunk_number"])):
        chunk = chunks[i]
        location = chunk["source_path"] + ("#" + chunk["anchor"] if chunk["anchor"] else "")
        candidates.append({**{k: v for k, v in chunk.items() if k != "tokens"},
                           "score": round(float(combined[i]), 6),
                           "lexical_score": round(float(lexical[i]), 6),
                           "semantic_score": round(float(semantic[i]), 6),
                           "location": location})
    return _select_results(candidates, top_k, per_section_limit=per_section_limit,
                           deduplicate_exact=deduplicate_exact)


def evaluate(query_file: Path = QUERY_PATH, index_path: Path = INDEX_PATH, *,
             per_section_limit: int | None = 2,
             deduplicate_exact: bool = False) -> dict[str, Any]:
    cases, ranking_k = _load_query_cases(query_file)
    if not ranking_k or any(k < 1 for k in ranking_k):
        raise ValueError("ranking_k debe contener enteros positivos")
    default_top_k = max(ranking_k)
    rows = []
    for case in cases:
        expected_zero = bool(case.get("expected_zero_results", False))
        top_k = max(default_top_k, int(case.get("top_k", default_top_k)))
        results = search(case["query"], top_k=top_k,
                         document_type=case.get("document_type"), topic=case.get("topic"), index_path=index_path,
                         per_section_limit=per_section_limit, deduplicate_exact=deduplicate_exact)
        retrieved_zero = not results
        zero_match = retrieved_zero == expected_zero
        positive = not expected_zero
        relevant_ranks = []
        if positive:
            expected_path = case["expected_path"]
            expected_section = case["expected_section"]
            expected_terms = case.get("expected_terms", [])
            for rank, result in enumerate(results, start=1):
                # Relevance evidence must all come from this one returned chunk.
                same_chunk = (result["source_path"] == expected_path
                              and expected_section.casefold() in result["section_path"].casefold()
                              and all(term.casefold() in result["text"].casefold() for term in expected_terms))
                if same_chunk:
                    relevant_ranks.append(rank)
            first_relevant_rank = relevant_ranks[0] if relevant_ranks else None
            recall_at_k = {str(k): int(any(rank <= k for rank in relevant_ranks)) for k in ranking_k}
            reciprocal_rank = 1 / first_relevant_rank if first_relevant_rank is not None else 0.0
            correct = first_relevant_rank is not None
        else:
            expected_path = ""
            expected_section = ""
            expected_terms = []
            first_relevant_rank = None
            recall_at_k = None
            reciprocal_rank = None
            correct = zero_match
        rows.append({"id": case["id"], "query": case["query"], "positive": positive,
                     "expected_zero_results": expected_zero, "expected_path": expected_path,
                     "expected_section": expected_section, "expected_terms": expected_terms,
                     "filters": {k: case[k] for k in ("document_type", "topic") if case.get(k)},
                     "correct": correct, "retrieved_zero_results": retrieved_zero,
                     "zero_results_match_expected": zero_match,
                     "relevant_ranks": relevant_ranks, "first_relevant_rank": first_relevant_rank,
                     "reciprocal_rank": reciprocal_rank, "recall_at_k": recall_at_k,
                     "observations": (f"Fragmento relevante en posición {first_relevant_rank} de {len(results)}."
                                      if positive and correct else
                                      "No apareció un fragmento que cumpla en conjunto ruta, sección y términos requeridos."
                                      if positive else
                                      "Filtro sin resultados, como se esperaba." if correct else
                                      "El filtro debía devolver cero resultados, pero devolvió fragmentos."),
                     "diversity_at_k": {
                         str(k): _diversity_counts(results[:k]) for k in ranking_k
                     },
                     "results": [{"rank": i, "chunk_id": r["chunk_id"], "path": r["source_path"], "section": r["section_path"],
                                  "score": r["score"], "location": r["location"],
                                  "relevant": i in relevant_ranks if positive else None}
                                 for i, r in enumerate(results, start=1)]})
    index = json.loads(index_path.read_text(encoding="utf-8"))
    positives = [row for row in rows if row["positive"]]
    negatives = [row for row in rows if not row["positive"]]
    ranking_metrics = {
        "positive_cases": len(positives),
        "recall_at_k": {str(k): sum(row["recall_at_k"][str(k)] for row in positives) / len(positives)
                         if positives else 0.0 for k in ranking_k},
        "mrr": sum(row["reciprocal_rank"] for row in positives) / len(positives) if positives else 0.0,
    }
    negative_metrics = {"negative_cases": len(negatives),
                        "zero_results_expected": sum(row["zero_results_match_expected"] for row in negatives),
                        "zero_results_accuracy": (sum(row["zero_results_match_expected"] for row in negatives) / len(negatives)
                                                  if negatives else None),
                        "returned_zero_results": sum(row["retrieved_zero_results"] for row in negatives)}
    positive_rows = [row for row in rows if row["positive"]]
    diversity_metrics = {
        str(k): {
            "positive_cases": len(positive_rows),
            "mean_distinct_sections": (sum(row["diversity_at_k"][str(k)]["distinct_sections"] for row in positive_rows) / len(positive_rows)
                                       if positive_rows else 0.0),
            "repeated_fragments": sum(row["diversity_at_k"][str(k)]["repeated_fragments"] for row in positive_rows),
        } for k in ranking_k
    }
    return {"configuration": _effective_configuration(index, query_file,
                                                         per_section_limit=per_section_limit,
                                                         deduplicate_exact=deduplicate_exact,
                                                         ranking_k=ranking_k),
            "index_documents": index["documents"],
            "index_chunks": index["chunk_count"],
            "ranking_k": ranking_k, "ranking_metrics": ranking_metrics,
            "negative_metrics": negative_metrics, "diversity_metrics": diversity_metrics,
            "passed": sum(row["correct"] for row in rows), "total": len(rows), "cases": rows}


def _diversity_counts(results: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter((result["source_path"], result["section_path"]) for result in results)
    return {"distinct_sections": len(counts),
            "repeated_fragments": sum(count - 1 for count in counts.values() if count > 1),
            "repeated_sections": [
                {"path": path, "section_path": section, "count": count}
                for (path, section), count in sorted(counts.items()) if count > 1
            ]}


def compare_reports(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    current_config = current["configuration"]
    baseline_config = baseline["configuration"]
    compatibility = {
        "same_queries": current_config.get("queries_sha256") == baseline_config.get("queries_sha256"),
        "same_corpus": current_config.get("corpus_manifest_sha256") == baseline_config.get("corpus_manifest_sha256"),
        "same_ranking_k": current.get("ranking_k") == baseline.get("ranking_k"),
        "same_configuration": current_config == baseline_config,
    }
    old_cases = {case["id"]: case for case in baseline["cases"]}
    new_cases = {case["id"]: case for case in current["cases"]}
    max_k = max(current.get("ranking_k", [1]))
    per_query = []
    for case_id in sorted(old_cases.keys() | new_cases.keys()):
        old, new = old_cases.get(case_id), new_cases.get(case_id)
        if old is None or new is None:
            per_query.append({"id": case_id, "status": "added" if old is None else "removed"})
            continue
        if old["positive"] != new["positive"] or old.get("expected_path") != new.get("expected_path") or old.get("expected_section") != new.get("expected_section") or old.get("expected_terms") != new.get("expected_terms") or old.get("filters") != new.get("filters"):
            definition_changed = True
        else:
            definition_changed = False
        old_rank, new_rank = old.get("first_relevant_rank"), new.get("first_relevant_rank")
        old_hit = old_rank is not None and old_rank <= max_k
        new_hit = new_rank is not None and new_rank <= max_k
        old_order = [result["chunk_id"] for result in old.get("results", [])]
        new_order = [result["chunk_id"] for result in new.get("results", [])]
        if not old["positive"]:
            per_query.append({"id": case_id, "positive": False,
                              "baseline_zero_results_match": old.get("zero_results_match_expected"),
                              "current_zero_results_match": new.get("zero_results_match_expected"),
                              "negative_filter_changed": old.get("zero_results_match_expected") != new.get("zero_results_match_expected"),
                              "definition_changed": definition_changed})
            continue
        recall_deltas = {str(k): new["recall_at_k"][str(k)] - old["recall_at_k"][str(k)]
                         for k in current["ranking_k"] if str(k) in old["recall_at_k"] and str(k) in new["recall_at_k"]}
        per_query.append({"id": case_id, "positive": True,
                          "baseline_relevant_ranks": old.get("relevant_ranks", []),
                          "current_relevant_ranks": new.get("relevant_ranks", []),
                          "baseline_first_relevant_rank": old_rank,
                          "current_first_relevant_rank": new_rank,
                          "first_rank_delta": new_rank - old_rank if old_rank is not None and new_rank is not None else None,
                          "baseline_hit_at_max_k": old_hit, "current_hit_at_max_k": new_hit,
                          "retrieval_changed": old_hit != new_hit,
                          "ranking_order_changed": old_order != new_order,
                          "reorder_only": old_hit == new_hit and old_order != new_order,
                          "recall_at_k_delta": recall_deltas,
                          "definition_changed": definition_changed})
    old_metrics, new_metrics = baseline["ranking_metrics"], current["ranking_metrics"]
    aggregate = {
        "recall_at_k_delta": {str(k): new_metrics["recall_at_k"].get(str(k), 0) - old_metrics["recall_at_k"].get(str(k), 0)
                               for k in current.get("ranking_k", [])},
        "mrr_delta": new_metrics["mrr"] - old_metrics["mrr"],
        "baseline_recall_at_k": old_metrics["recall_at_k"],
        "current_recall_at_k": new_metrics["recall_at_k"],
        "baseline_mrr": old_metrics["mrr"], "current_mrr": new_metrics["mrr"],
        "retrieval_changes": sum(case.get("retrieval_changed", False) for case in per_query),
        "reorder_only_changes": sum(case.get("reorder_only", False) for case in per_query),
        "positive_cases": new_metrics["positive_cases"],
    }
    return {"compatibility": compatibility, "warnings": [key for key, value in compatibility.items() if not value],
            "aggregate": aggregate,
            "negative_filter_comparison": [case for case in per_query if case.get("positive") is False],
            "per_query": per_query}


def main() -> int:
    parser = argparse.ArgumentParser(description="Retriever híbrido local de VERONICA")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build", help="Reconstruye el índice desde knowledge/ y playbook(s)/")
    search_parser = commands.add_parser("search", help="Busca fragmentos con procedencia")
    search_parser.add_argument("query")
    search_parser.add_argument("--top-k", type=int, default=5)
    search_parser.add_argument("--type", choices=("knowledge", "playbook"))
    search_parser.add_argument("--topic")
    search_parser.add_argument("--per-section-limit", type=int, default=2,
                               help="Máximo de fragmentos por sección (0 desactiva el límite; predeterminado: 2)")
    search_parser.add_argument("--deduplicate-exact", action="store_true",
                               help="Omite textos idénticos exactos")
    evaluate_parser = commands.add_parser("evaluate", help="Ejecuta preguntas de referencia reproducibles")
    evaluate_parser.add_argument("--per-section-limit", type=int, default=2,
                                 help="Máximo de fragmentos por sección (0 desactiva el límite; predeterminado: 2)")
    evaluate_parser.add_argument("--deduplicate-exact", action="store_true",
                                 help="Omite textos idénticos exactos")
    baseline_parser = commands.add_parser("baseline", help="Actualiza explícitamente la referencia versionada")
    baseline_parser.add_argument("--update", action="store_true", required=True,
                                 help=f"Reemplaza deliberadamente {BASELINE_PATH.name}")
    compare_parser = commands.add_parser("compare", help="Compara una evaluación actual con la referencia versionada")
    compare_parser.add_argument("--per-section-limit", type=int, default=2)
    compare_parser.add_argument("--deduplicate-exact", action="store_true")
    args = parser.parse_args()
    if args.command == "build":
        index = build_index()
        print(f"Índice regenerado: {index['documents']} documentos, {index['chunk_count']} fragmentos en {INDEX_PATH.relative_to(ROOT)}")
        return 0
    if args.command == "search":
        results = search(args.query, top_k=max(1, args.top_k), document_type=args.type, topic=args.topic,
                         per_section_limit=args.per_section_limit or None,
                         deduplicate_exact=args.deduplicate_exact)
        for result in results:
            print(f"[{result['score']:.4f} | léxico {result['lexical_score']:.4f} | semántico {result['semantic_score']:.4f}] "
                  f"{result['location']} — {result['document_title']} > {result['section_path']}\n{result['text']}\n")
        if not results:
            print("Sin resultados para esos filtros.")
        return 0
    if args.command == "baseline":
        report = evaluate()
        BASELINE_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        metrics = report["ranking_metrics"]
        print(f"Referencia actualizada explícitamente: {BASELINE_PATH.relative_to(ROOT)}")
        print("Recall@k: " + ", ".join(f"{k}={value:.3f}" for k, value in metrics["recall_at_k"].items()))
        print(f"MRR: {metrics['mrr']:.3f}; positivos: {metrics['positive_cases']}; consultas: {report['total']}")
        print(f"Expectativas acertadas: {report['passed']}/{report['total']}")
        return 0
    if args.command == "compare":
        if not BASELINE_PATH.exists():
            parser.error(f"No existe {BASELINE_PATH.name}; genere una referencia intencionalmente con: python rag_mvp.py baseline --update")
        limit = args.per_section_limit or None
        current = evaluate(per_section_limit=limit, deduplicate_exact=args.deduplicate_exact)
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        comparison = compare_reports(current, baseline)
        comparison_path = ROOT / ".veronica" / "retrieval_comparison.json"
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Comparación guardada: {comparison_path.relative_to(ROOT)}")
        print("Compatibilidad: " + ", ".join(f"{key}={'sí' if value else 'NO'}" for key, value in comparison["compatibility"].items()))
        if comparison["warnings"]:
            print("AVISO: comparación no equivalente en: " + ", ".join(comparison["warnings"]))
        aggregate = comparison["aggregate"]
        for k, delta in aggregate["recall_at_k_delta"].items():
            print(f"Recall@{k}: {aggregate['baseline_recall_at_k'].get(k, 0):.3f} -> {aggregate['current_recall_at_k'].get(k, 0):.3f} ({delta:+.3f})")
        print(f"MRR: {aggregate['baseline_mrr']:.3f} -> {aggregate['current_mrr']:.3f} ({aggregate['mrr_delta']:+.3f}); "
              f"cambios de recuperación={aggregate['retrieval_changes']}, solo reordenamientos={aggregate['reorder_only_changes']}")
        for case in comparison["per_query"]:
            if case.get("positive"):
                print(f"{case['id']}: relevancia {case['baseline_relevant_ranks']} -> {case['current_relevant_ranks']}; "
                      f"Recall@k delta={case['recall_at_k_delta']}; recuperación cambió={case['retrieval_changed']}; "
                      f"solo orden={case['reorder_only']}")
            elif case.get("positive") is False:
                print(f"{case['id']} [filtro negativo]: cero resultados correcto "
                      f"{case['baseline_zero_results_match']} -> {case['current_zero_results_match']}")
        return 0
    per_section_limit = args.per_section_limit or None
    report = evaluate(per_section_limit=per_section_limit,
                      deduplicate_exact=args.deduplicate_exact)
    comparison_specs = [
        ("sin límite", None, False),
        ("deduplicación exacta", None, True),
        ("límite 1", 1, False),
        ("límite 2", 2, False),
    ]
    comparisons = {}
    for name, limit, deduplicate in comparison_specs:
        alternative = evaluate(per_section_limit=limit, deduplicate_exact=deduplicate)
        comparisons[name] = {
            "configuration": alternative["configuration"],
            "ranking_metrics": alternative["ranking_metrics"],
            "diversity_metrics": alternative["diversity_metrics"],
            "per_query": {
                case["id"]: {
                    "first_relevant_rank": case["first_relevant_rank"],
                    "recall_at_k": case["recall_at_k"],
                    "reciprocal_rank": case["reciprocal_rank"],
                    "diversity_at_k": case["diversity_at_k"],
                } for case in alternative["cases"]
            },
        }
    report["policy_comparison"] = comparisons
    baseline_path = ROOT / ".veronica" / "retrieval_evaluation_baseline.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        report["baseline_repetition_audit"] = baseline.get("repetition_audit", {})
    report_path = ROOT / ".veronica" / "retrieval_evaluation.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    metric = report["ranking_metrics"]
    negative = report["negative_metrics"]
    lines = ["# Resultados de consultas de referencia", "",
             f"**Diversidad aplicada:** límite de {per_section_limit or 'ninguno'} fragmentos por sección; "
             f"deduplicación exacta: {'sí' if args.deduplicate_exact else 'no'}.", "",
             f"Recuperación correcta: **{report['passed']}/{report['total']}**.", "",
             f"**Recall@k (casos positivos, k={', '.join(map(str, report['ranking_k']))}):** "
             + " · ".join(f"Recall@{k} = {value:.3f}" for k, value in metric["recall_at_k"].items()),
             f"**MRR (casos positivos):** {metric['mrr']:.3f} · n={metric['positive_cases']}",
             f"**Filtros negativos:** {negative['zero_results_expected']}/{negative['negative_cases']} "
             f"coinciden con cero resultados (accuracy={negative['zero_results_accuracy']}).",
             "**Diversidad top-10 (casos positivos):** "
             f"{report['diversity_metrics']['10']['mean_distinct_sections']:.3f} secciones distintas de media; "
             f"{report['diversity_metrics']['10']['repeated_fragments']} fragmentos repetidos por sección.",
             "", "## Comparación de políticas", "",
             "| Política | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | Repetidos top-10 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, result in comparisons.items():
        m = result["ranking_metrics"]
        d = result["diversity_metrics"]["10"]
        lines.append(f"| {name} | {m['recall_at_k']['1']:.3f} | {m['recall_at_k']['3']:.3f} | "
                     f"{m['recall_at_k']['5']:.3f} | {m['recall_at_k']['10']:.3f} | "
                     f"{m['mrr']:.3f} | {d['repeated_fragments']} |")
    lines += ["", "Las métricas de diversidad por consulta y los rankings de cada política están en el JSON.", "",
             "La puntuación ordena resultados y no representa una probabilidad de confianza.", ""]
    for case in report["cases"]:
        lines += [f"## {case['id']}", "", f"- **Consulta:** {case['query']}",
                  f"- **Esperado:** `{case['expected_path']}` · {case['expected_section']}",
                  f"- **Conceptos/términos:** {', '.join(case['expected_terms']) or 'sin criterio léxico explícito'}",
                  f"- **Filtros:** {json.dumps(case['filters'], ensure_ascii=False) if case['filters'] else 'ninguno'}",
                  f"- **Tipo:** {'positivo' if case['positive'] else 'negativo (cero resultados esperado)'}",
                  f"- **Correcto:** {'sí' if case['correct'] else 'no'}",
                  f"- **Cero resultados:** {'sí' if case['retrieved_zero_results'] else 'no'}; coincide con expectativa: {'sí' if case['zero_results_match_expected'] else 'no'}",
                  f"- **Recall@k:** {json.dumps(case['recall_at_k'], ensure_ascii=False) if case['positive'] else 'no aplica'}",
                  f"- **Primera posición relevante / RR:** {case['first_relevant_rank'] if case['positive'] else 'no aplica'} / {case['reciprocal_rank'] if case['positive'] else 'no aplica'}",
                  f"- **Diversidad top-10:** {case['diversity_at_k']['10']['distinct_sections']} secciones distintas; "
                  f"{case['diversity_at_k']['10']['repeated_fragments']} fragmentos repetidos por sección.",
                  f"- **Observación:** {case['observations']}",
                  "- **Resultados:**"]
        lines.extend(f"  - #{result['rank']} {'[relevante] ' if result['relevant'] else ''}`{result['location']}` — `{result['score']:.4f}`; sección: {result['section']}; id: `{result['chunk_id']}`"
                     for result in case["results"])
        lines.append("")
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        lines += ["## Auditoría de repeticiones de la línea base", "",
                  "Se compararon los textos completos de los fragmentos listados, junto con sus rutas, rutas de sección, IDs y puntuaciones. "
                  "En estos primeros 10 resultados no hay duplicados exactos ni casi duplicados; son fragmentos diferentes con contenido complementario. "
                  "Las continuaciones de tablas, pseudocódigo o procedimientos aportan contenido adicional y no se consideran duplicados.", ""]
        for case in baseline["cases"]:
            groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
            for result in case.get("detailed_results", []):
                groups[(result["path"], result["section_path"])].append(result)
            for (path, section), results in groups.items():
                if len(results) > 1:
                    chunks = ", ".join(f"#{result['rank']} `{result['chunk_id']}` ({result['score']:.4f})"
                                        for result in results)
                    lines.append(f"- **{case['id']}** · `{path}` · {section}: {chunks}.")
        lines.append("")
    (ROOT / ".veronica" / "retrieval_evaluation.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Consultas correctas: {report['passed']}/{report['total']} — {report_path.relative_to(ROOT)}")
    print("Recall@k: " + ", ".join(f"{k}={value:.3f}" for k, value in report["ranking_metrics"]["recall_at_k"].items()))
    print(f"MRR: {report['ranking_metrics']['mrr']:.3f} sobre {report['ranking_metrics']['positive_cases']} casos positivos")
    negatives = report["negative_metrics"]
    print(f"Filtros negativos: {negatives['zero_results_expected']}/{negatives['negative_cases']} coinciden con cero resultados")
    for case in report["cases"]:
        first = case["results"][0]["location"] if case["results"] else "sin resultados"
        print(f"{'OK' if case['correct'] else 'FALLO'} {case['id']}: esperado {case['expected_path']}; primero {first}")
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
