#!/usr/bin/env python3
"""Audit local Markdown references and repository document metadata."""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parent
PENDING_PLAYBOOKS = (
    "playbook/classification.md",
    "playbook/ml_workflow.md",
    "playbook/regression.md",
    "playbook/time_series.md",
)
LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
HEADING_RE = re.compile(r"^ {0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
CITATION_RE = re.compile(
    r"(?P<author>Géron|Müller(?: y Guido| y Sarah Guido)?|Molin|McKinney|ISLP|James et al\.?|ML Design Patterns|Patterns)"
    r"[^\n\]]{0,120}?\b(?:cap(?:ítulo)?\.?|ch\.?)\s*(?P<chapter>\d+)"
    r"(?P<tail>[^\n\];]{0,100})",
    re.IGNORECASE,
)
AUTHOR_FOLDERS = {
    "géron": "geron_hands_on_ml",
    "müller": "muller_introduction_ml_python",
    "molin": "molin_hands_on_pandas",
    "mckinney": "mckinney_python_data_analysis",
    "islp": "islp_python",
    "james et al.": "islp_python",
    "ml design patterns": "ml_design_patterns",
    "patterns": "ml_design_patterns",
}


@dataclass
class Diagnostic:
    kind: str
    location: str
    message: str
    error: bool = True


def frontmatter(text: str) -> tuple[dict[str, object], str]:
    """Parse the small YAML subset used in this corpus without dependencies."""
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end < 0:
        return {}, text
    raw = text[4:end]
    result: dict[str, object] = {}
    current_list: str | None = None
    for line in raw.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        list_item = re.match(r"^\s+-\s+(.*)$", line)
        if list_item and current_list:
            result[current_list].append(_yaml_value(list_item.group(1)))  # type: ignore[union-attr]
            continue
        match = re.match(r"^([A-Za-z_][\w-]*):(?:\s*(.*))?$", line)
        if not match:
            current_list = None
            continue
        key, value = match.group(1), (match.group(2) or "").strip()
        if not value:
            result[key] = []
            current_list = key
        else:
            result[key] = _yaml_value(value)
            current_list = None
    return result, text[end + 4 :].lstrip("\n")


def _yaml_value(value: str) -> object:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    if value in ("true", "false"):
        return value == "true"
    if value == "null":
        return None
    return value


def slugify(heading: str) -> str:
    heading = re.sub(r"[`*_]", "", heading).lower().strip()
    heading = "".join(char for char in heading if char in "- " or unicodedata.category(char)[0] in "LN")
    return re.sub(r"\s+", "-", heading)


def anchors(text: str) -> set[str]:
    found: set[str] = set()
    counts: dict[str, int] = {}
    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if not match:
            continue
        slug = slugify(match.group(2))
        count = counts.get(slug, 0)
        found.add(slug if count == 0 else f"{slug}-{count}")
        counts[slug] = count + 1
    return found


def numbered_headings(text: str) -> set[str]:
    numbers: set[str] = set()
    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if not match:
            continue
        number = re.match(r"(?:Pattern\s+)?(\d+(?:\.\d+)*)\b", match.group(2), re.I)
        if number:
            numbers.add(number.group(1))
    return numbers


def chapter_path(author: str, chapter: int) -> Path | None:
    folder = AUTHOR_FOLDERS.get(author.casefold())
    if not folder:
        return None
    matches = sorted((ROOT / "sources" / folder).glob(f"ch{chapter:02d}_*.md"))
    return matches[0] if len(matches) == 1 else None


def markdown_files(folder: str) -> list[Path]:
    return sorted((ROOT / folder).rglob("*.md"))


def audit() -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    knowledge_files = markdown_files("knowledge")
    for source in knowledge_files:
        text = source.read_text(encoding="utf-8")
        for match in LINK_RE.finditer(text):
            target = match.group(1).strip().split()[0].strip("<>")
            if not target or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target) or target.startswith("//"):
                continue
            path_part, _, fragment = target.partition("#")
            path_part = unquote(path_part)
            if not path_part:
                dest = source
            else:
                dest = (source.parent / path_part).resolve()
            try:
                dest.relative_to(ROOT)
            except ValueError:
                diagnostics.append(Diagnostic("RUTA", _loc(source, text, match.start()), f"destino fuera del proyecto: {target}"))
                continue
            if not dest.is_file():
                diagnostics.append(Diagnostic("ROTA", _loc(source, text, match.start()), f"no existe el destino local: {target}"))
                continue
            if fragment:
                available = anchors(dest.read_text(encoding="utf-8")) if dest.suffix.lower() == ".md" else set()
                if unquote(fragment).lower() not in available:
                    diagnostics.append(Diagnostic("ANCLA", _loc(source, text, match.start()), f"no existe #{fragment} en {dest.relative_to(ROOT).as_posix()}"))
            elif dest.is_file() and dest.suffix.lower() == ".md" and dest.is_relative_to(ROOT / "sources"):
                diagnostics.append(Diagnostic("AMBIGUA", _loc(source, text, match.start()), f"cita el archivo completo {dest.relative_to(ROOT).as_posix()} sin sección; comprobar el contexto", False))

        # Validate explicit chapter/section citations written as prose (for
        # example, “Géron, cap. 3, §§2.2 y 8”) against source headings.
        for citation in CITATION_RE.finditer(text):
            tail = citation.group("tail")
            section_marker = re.search(r"§{1,2}\s*", tail)
            if not section_marker:
                continue
            author = citation.group("author").casefold()
            if author.startswith("müller"):
                author = "müller"
            elif author.startswith("james"):
                author = "james et al."
            source_doc = chapter_path(author, int(citation.group("chapter")))
            if not source_doc or not source_doc.is_file():
                diagnostics.append(Diagnostic("CITA", _loc(source, text, citation.start()), f"no se pudo resolver {citation.group('author')}, capítulo {citation.group('chapter')}"))
                continue
            section_text = tail[section_marker.end():]
            cited_sections = re.findall(r"\d+(?:\.\d+)*", section_text)
            available_sections = numbered_headings(source_doc.read_text(encoding="utf-8"))
            for number in cited_sections:
                if number not in available_sections:
                    diagnostics.append(Diagnostic("SECCIÓN CITADA", _loc(source, text, citation.start()), f"{citation.group('author')}, cap. {citation.group('chapter')}, §{number} no aparece en {source_doc.relative_to(ROOT).as_posix()}"))

        meta, _ = frontmatter(text)
        area = source.parent.name
        if not meta:
            diagnostics.append(Diagnostic("METADATOS", source.relative_to(ROOT).as_posix(), "falta front matter"))
            continue
        if meta.get("knowledge_area") != area:
            diagnostics.append(Diagnostic("METADATOS", source.relative_to(ROOT).as_posix(), f"knowledge_area={meta.get('knowledge_area')!r} no coincide con carpeta {area!r}"))
        status = meta.get("status")
        if not isinstance(status, str) or not status.strip():
            diagnostics.append(Diagnostic("METADATOS", source.relative_to(ROOT).as_posix(), "status ausente o vacío"))
        for field in ("created_from", "target_use"):
            value = meta.get(field)
            if value is None:
                diagnostics.append(Diagnostic("METADATOS", source.relative_to(ROOT).as_posix(), f"falta {field}"))
                continue
            values = value if isinstance(value, list) else [value]
            for item in values:
                if not isinstance(item, str):
                    continue
                path = item.rstrip("/")
                if path.startswith("sources/") or path.startswith("knowledge/") or path.startswith("playbook/"):
                    if not (ROOT / path).exists():
                        diagnostics.append(Diagnostic("METADATOS", source.relative_to(ROOT).as_posix(), f"{field} apunta a destino inexistente: {item}"))

    # Source metadata names intended knowledge destinations; missing files are
    # reported separately because the README describes them as proposed mappings.
    for source in (p for p in markdown_files("sources") if p.name not in {"README.md", "source_catalog.md"}):
        meta, _ = frontmatter(source.read_text(encoding="utf-8"))
        if not meta:
            diagnostics.append(Diagnostic("METADATOS", source.relative_to(ROOT).as_posix(), "falta front matter"))
            continue
        for field in ("title", "topics", "target_knowledge_files"):
            if field not in meta:
                diagnostics.append(Diagnostic("METADATOS", source.relative_to(ROOT).as_posix(), f"falta {field}"))
        destinations = meta.get("target_knowledge_files", [])
        if isinstance(destinations, list):
            for target in destinations:
                if isinstance(target, str) and target.startswith("knowledge/") and not (ROOT / target).is_file():
                    diagnostics.append(Diagnostic("DESTINO PROPUESTO", source.relative_to(ROOT).as_posix(), f"{target} aún no existe; revisar si es un mapeo futuro", False))

    # Catalog file links and its aggregate status counts must agree with files.
    catalog = ROOT / "sources" / "source_catalog.md"
    catalog_text = catalog.read_text(encoding="utf-8") if catalog.exists() else ""
    for match in LINK_RE.finditer(catalog_text):
        target = match.group(1).strip().split()[0].strip("<>")
        if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target) or target.startswith("//"):
            continue
        path_part = unquote(target.split("#", 1)[0])
        if path_part and not (catalog.parent / path_part).is_file():
            diagnostics.append(Diagnostic("CATÁLOGO", _loc(catalog, catalog_text, match.start()), f"enlace a archivo inexistente: {target}"))
    records = [frontmatter(p.read_text(encoding="utf-8"))[0] for p in markdown_files("sources") if p.name not in {"README.md", "source_catalog.md"}]
    statuses = Counter()
    for status in re.findall(r"\|\s*`(complete|partial|needs_review|empty)`\s*\|", catalog_text):
        statuses[status] += 1
    expected = Counter({"complete": 0, "partial": 0, "needs_review": 0, "empty": 0})
    for row in re.findall(r"\|\s*\[([^\]]+)\]\(([^)]+\.md)\)\s*\|\s*`(complete|partial|needs_review|empty)`\s*\|", catalog_text):
        expected[row[2]] += 1
    for status, count in expected.items():
        stated = re.search(rf"- `{status}`:\s*\*\*(\d+)\*\*", catalog_text)
        if stated and int(stated.group(1)) != count:
            diagnostics.append(Diagnostic("CATÁLOGO", "sources/source_catalog.md", f"total {status} declarado {stated.group(1)}; filas catalogadas {count}"))
    catalog_map = re.search(r"## 4\. Mapeo de fuentes hacia `knowledge/`(.*?)(?=\n## |\Z)", catalog_text, re.S)
    if catalog_map:
        for target in re.findall(r"`(knowledge/[^`]+\.md)`", catalog_map.group(1)):
            if not (ROOT / target).is_file():
                diagnostics.append(Diagnostic("CATÁLOGO", "sources/source_catalog.md", f"mapeo activo a knowledge inexistente: {target}"))
    listed = re.findall(r"\|\s*\[[^\]]+\]\(([^)]+\.md)\)\s*\|\s*`(?:complete|partial|needs_review|empty)`\s*\|", catalog_text)
    listed_sources = {Path(path).as_posix() for path in listed}
    actual_sources = {p.relative_to(ROOT / "sources").as_posix() for p in markdown_files("sources") if p.name not in {"README.md", "source_catalog.md"}}
    for path in sorted(actual_sources - listed_sources):
        diagnostics.append(Diagnostic("CATÁLOGO", "sources/source_catalog.md", f"fuente no catalogada: {path}"))
    for path in sorted(listed_sources - actual_sources):
        diagnostics.append(Diagnostic("CATÁLOGO", "sources/source_catalog.md", f"fuente catalogada inexistente: {path}"))

    if (ROOT / "playbook").is_dir():
        inventory_docs = [ROOT / "sources" / "README.md", catalog, ROOT / "RAG_MVP.md"]
        for relative in PENDING_PLAYBOOKS:
            pending = ROOT / relative
            if not pending.is_file():
                diagnostics.append(Diagnostic("PLAYBOOK", relative, "archivo pendiente no existe"))
            elif pending.stat().st_size:
                diagnostics.append(Diagnostic("PLAYBOOK", relative, "está registrado como pendiente, pero contiene datos"))
            for document in inventory_docs:
                if not document.is_file() or relative not in document.read_text(encoding="utf-8"):
                    diagnostics.append(Diagnostic("PLAYBOOK", relative, f"no está registrado en {document.relative_to(ROOT).as_posix()}"))
    return diagnostics


def _loc(path: Path, text: str, offset: int) -> str:
    return f"{path.relative_to(ROOT).as_posix()}:{text.count(chr(10), 0, offset) + 1}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="raíz del proyecto (para uso desde otras carpetas)")
    parser.add_argument("--verbose", action="store_true", help="mostrar cada referencia sin ancla y cada destino propuesto ausente")
    args = parser.parse_args()
    global ROOT
    if args.root:
        ROOT = args.root.resolve()
    diagnostics = audit()
    details = diagnostics if args.verbose else [d for d in diagnostics if d.error]
    for item in details:
        level = "ERROR" if item.error else ("AMBIGUA" if item.kind == "AMBIGUA" else "INFO")
        print(f"{level} [{item.kind}] {item.location}: {item.message}")
    errors = sum(d.error for d in diagnostics)
    ambiguities = sum(d.kind == "AMBIGUA" for d in diagnostics)
    proposed = sum(d.kind == "DESTINO PROPUESTO" for d in diagnostics)
    if not args.verbose:
        by_document = Counter(d.location.split(":", 1)[0] for d in diagnostics if d.kind == "AMBIGUA")
        for document, count in sorted(by_document.items()):
            print(f"AMBIGUAS [{document}]: {count} referencia(s) a archivo(s) completo(s), sin ancla de sección.")
    print(f"Resumen: {errors} error(es), {ambiguities} referencia(s) ambigua(s), {proposed} destino(s) propuesto(s) ausente(s).")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
