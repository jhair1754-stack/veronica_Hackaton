"""Pruebas aisladas para el retriever RAG local."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import rag_mvp


class MarkdownChunkingTests(unittest.TestCase):
    def test_heading_paths_and_content_are_preserved(self) -> None:
        text = """# Raíz
Texto de raíz.

## Rama
Texto de rama.

### Hoja
Texto de hoja.
"""

        self.assertEqual(
            rag_mvp._markdown_blocks(text),
            [
                ("Raíz", "Texto de raíz."),
                ("Raíz\nRama", "Texto de rama."),
                ("Raíz\nRama\nHoja", "Texto de hoja."),
            ],
        )

    def test_long_content_is_split_within_limit_when_possible(self) -> None:
        block = "\n".join(f"line{i} " + "x" * 100 for i in range(20))
        pieces = rag_mvp._split_long(block)

        self.assertGreater(len(pieces), 1)
        self.assertTrue(all(len(piece) <= rag_mvp.MAX_CHUNK_CHARS for piece in pieces))
        self.assertEqual(" ".join(block.split()), " ".join(" ".join(pieces).split()))

    def test_fenced_code_heading_is_not_a_markdown_heading(self) -> None:
        blocks = rag_mvp._markdown_blocks(
            "# Real\nTexto.\n\n```python\n## No es encabezado\nprint('ok')\n```\n\n## Siguiente\nFin."
        )

        self.assertEqual(len(blocks), 3)
        self.assertEqual(blocks[0], ("Real", "Texto."))
        self.assertEqual(blocks[1][0], "Real")
        self.assertIn("## No es encabezado", blocks[1][1])
        self.assertEqual(blocks[2], ("Real\nSiguiente", "Fin."))


class SpanishMorphologyTests(unittest.TestCase):
    def test_action_nouns_and_infinitives_share_a_lexical_key(self) -> None:
        self.assertEqual(rag_mvp.tokenize("eliminar interpolar"),
                         rag_mvp.tokenize("eliminación interpolación"))
        self.assertEqual(rag_mvp.tokenize("eliminar interpolar"),
                         rag_mvp.tokenize("eliminaciones interpolaciones"))

    def test_unrelated_words_keep_distinct_lexical_keys(self) -> None:
        self.assertNotEqual(rag_mvp.tokenize("eliminar"), rag_mvp.tokenize("elixir"))
        self.assertNotEqual(rag_mvp.tokenize("interpolar"), rag_mvp.tokenize("interior"))

    def test_missing_values_paraphrase_retrieves_the_expected_chunk(self) -> None:
        index_path = Path(rag_mvp.ROOT) / ".veronica" / "rag_index.json"
        rag_mvp.build_index(index_path)

        results = rag_mvp.search(
            "¿Cómo identificar valores incompletos y qué alternativas como eliminar o interpolar existen?",
            top_k=10,
            topic="preprocessing",
            index_path=index_path,
        )

        relevant = [result for result in results
                    if result["source_path"] == "knowledge/preprocessing/data_cleaning.md"
                    and "4.1 Valores faltantes" in result["section_path"]
                    and all(term in result["text"].casefold() for term in ("eliminación", "interpolación"))]
        self.assertTrue(relevant, "se esperaba el fragmento de 4.1 con ambos términos literales")


class MetadataAndLoadingTests(unittest.TestCase):
    def test_frontmatter_extracts_scalars_and_returns_body(self) -> None:
        text = '---\ntitle: "Título de prueba"\ntopic: demo\ncount: 3\n---\n\n# Encabezado\nCuerpo.'

        metadata, body = rag_mvp._frontmatter(text)

        self.assertEqual(metadata, {"title": "Título de prueba", "topic": "demo", "count": "3"})
        self.assertEqual(body, "# Encabezado\nCuerpo.")

    def test_frontmatter_handles_missing_incomplete_and_ignored_values(self) -> None:
        plain = "# Texto\nCuerpo"
        self.assertEqual(rag_mvp._frontmatter(plain), ({}, plain))

        incomplete = "---\ntitle: sin cierre\n# Cuerpo"
        self.assertEqual(rag_mvp._frontmatter(incomplete), ({}, incomplete))

        metadata, body = rag_mvp._frontmatter(
            "---\nnull_value: null\nnone_value: None\nunknown_value: unknown\n"
            "items:\n  - uno\ntitle: Válido\n---\nCuerpo"
        )
        self.assertEqual(metadata, {"title": "Válido"})
        self.assertEqual(body, "Cuerpo")

    def test_load_documents_applies_title_topic_priority_and_folder_type(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = {
                "knowledge/one.md": "---\ntitle: Título YAML\nknowledge_area: área-principal\n"
                "topic: tema-secundario\nsubject: asunto\n---\n# Título H1\nTexto uno.",
                "playbook/two.md": "---\ntopic: topic-playbook\n---\n# Playbook H1\nTexto dos.",
                "playbooks/three.md": "---\nsubject: subject-playbooks\n---\n# Playbooks H1\nTexto tres.",
            }
            for relative, content in files.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")

            with patch.object(rag_mvp, "ROOT", root):
                docs = rag_mvp.load_documents()

        by_path = {doc["path"]: doc for doc in docs}
        knowledge = by_path["knowledge/one.md"]["chunks"][0]
        self.assertEqual(knowledge["document_title"], "Título YAML")
        self.assertEqual(knowledge["topic"], "área-principal")
        self.assertEqual(knowledge["document_type"], "knowledge")
        self.assertEqual(by_path["playbook/two.md"]["chunks"][0]["document_title"], "Playbook H1")
        self.assertEqual(by_path["playbook/two.md"]["chunks"][0]["topic"], "topic-playbook")
        self.assertEqual(by_path["playbook/two.md"]["chunks"][0]["document_type"], "playbook")
        self.assertEqual(by_path["playbooks/three.md"]["chunks"][0]["topic"], "subject-playbooks")
        self.assertEqual(by_path["playbooks/three.md"]["chunks"][0]["document_type"], "playbook")

    def test_empty_and_heading_only_files_produce_no_indexable_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, content in {
                "empty.md": "",
                "metadata.md": "---\ntitle: Solo metadata\n---\n",
                "headings.md": "---\ntopic: demo\n---\n# Encabezado\n## Subencabezado",
            }.items():
                path = root / "knowledge" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            index_path = root / "index.json"

            with patch.object(rag_mvp, "ROOT", root):
                self.assertEqual(rag_mvp.load_documents(), [])
                index = rag_mvp.build_index(index_path)

        self.assertEqual(index["documents"], 0)
        self.assertEqual(index["chunk_count"], 0)
        self.assertEqual(index["chunks"], [])


class SearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "knowledge").mkdir()
        (self.root / "playbook").mkdir()
        (self.root / "knowledge" / "alpha.md").write_text(
            "---\ntopic: alpha\n---\n# Alpha\nsharedterm alphaterm.", encoding="utf-8"
        )
        (self.root / "playbook" / "beta.md").write_text(
            "---\ntopic: beta\n---\n# Beta\nsharedterm betaterm.", encoding="utf-8"
        )
        self.index_path = self.root / "index.json"
        self.root_patch = patch.object(rag_mvp, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.addCleanup(self.temporary.cleanup)
        rag_mvp.build_index(self.index_path)

    def test_document_type_and_topic_filters_combine(self) -> None:
        self.assertEqual(
            {r["document_type"] for r in rag_mvp.search("sharedterm", document_type="knowledge", index_path=self.index_path)},
            {"knowledge"},
        )
        self.assertEqual(
            {r["topic"] for r in rag_mvp.search("sharedterm", topic="beta", index_path=self.index_path)},
            {"beta"},
        )
        combined = rag_mvp.search(
            "sharedterm", document_type="knowledge", topic="alpha", index_path=self.index_path
        )
        self.assertEqual([r["source_path"] for r in combined], ["knowledge/alpha.md"])
        self.assertEqual(rag_mvp.search("sharedterm", topic="missing", index_path=self.index_path), [])
        self.assertEqual(rag_mvp.search("sharedterm", topic="", document_type="missing", index_path=self.index_path), [])

    def test_query_with_no_retrievable_terms_returns_no_results(self) -> None:
        results = rag_mvp.search("unlistedquasar nebularquartz", index_path=self.index_path)
        self.assertEqual(results, [])

    def test_per_section_limit_preserves_rank_order_of_kept_results(self) -> None:
        candidates = [
            {"source_path": "knowledge/a.md", "section_path": "A", "text": "first"},
            {"source_path": "knowledge/a.md", "section_path": "A", "text": "second"},
            {"source_path": "knowledge/a.md", "section_path": "B", "text": "third"},
            {"source_path": "knowledge/a.md", "section_path": "A", "text": "fourth"},
        ]

        kept = rag_mvp._select_results(candidates, 3, per_section_limit=2)

        self.assertEqual([result["text"] for result in kept], ["first", "second", "third"])

    def test_section_identity_includes_document_path_and_section_path(self) -> None:
        candidates = [
            {"source_path": "knowledge/a.md", "section_path": "Shared", "text": "a1"},
            {"source_path": "knowledge/a.md", "section_path": "Other", "text": "a2"},
            {"source_path": "knowledge/b.md", "section_path": "Shared", "text": "b1"},
        ]

        kept = rag_mvp._select_results(candidates, 3, per_section_limit=1)

        self.assertEqual([result["text"] for result in kept], ["a1", "a2", "b1"])

    def test_returns_all_results_when_fewer_sections_are_available(self) -> None:
        candidates = [
            {"source_path": "knowledge/a.md", "section_path": "Only", "text": "first"},
            {"source_path": "knowledge/a.md", "section_path": "Only", "text": "second"},
        ]

        kept = rag_mvp._select_results(candidates, 5, per_section_limit=2)

        self.assertEqual(kept, candidates)

    def test_exact_deduplication_drops_only_identical_text(self) -> None:
        candidates = [
            {"source_path": "knowledge/a.md", "section_path": "A", "text": "same"},
            {"source_path": "knowledge/b.md", "section_path": "B", "text": "same"},
            {"source_path": "knowledge/b.md", "section_path": "B", "text": "complementary"},
        ]

        kept = rag_mvp._select_results(candidates, 3, deduplicate_exact=True)

        self.assertEqual([result["text"] for result in kept], ["same", "complementary"])

    def test_stable_ids_repeat_change_with_content_and_ignore_processing_order(self) -> None:
        first_index = rag_mvp.build_index(self.index_path)
        second_index = rag_mvp.build_index(self.index_path)
        first_ids = {chunk["chunk_id"] for chunk in first_index["chunks"]}
        second_ids = {chunk["chunk_id"] for chunk in second_index["chunks"]}
        self.assertEqual(first_ids, second_ids)

        original_chunks = {
            (chunk["source_path"], chunk["section_path"], chunk["text"]): chunk["chunk_id"]
            for chunk in first_index["chunks"]
        }
        original_alpha_text = (self.root / "knowledge" / "alpha.md").read_text(encoding="utf-8")
        (self.root / "knowledge" / "alpha.md").write_text(
            "---\ntopic: alpha\n---\n# Alpha\nchangedterm alphaterm.", encoding="utf-8"
        )
        changed_chunks = [chunk for doc in rag_mvp.load_documents() for chunk in doc["chunks"]]
        changed = next(c for c in changed_chunks if c["source_path"] == "knowledge/alpha.md")
        original = next(chunk for chunk in first_index["chunks"] if chunk["source_path"] == "knowledge/alpha.md")
        self.assertNotEqual(changed["chunk_id"], original["chunk_id"])

        # File creation order differs in this corpus; IDs still depend only on path, section, and content.
        with tempfile.TemporaryDirectory() as other_temporary:
            other_root = Path(other_temporary)
            for relative in ("playbook/beta.md", "knowledge/alpha.md"):
                destination = other_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                content = original_alpha_text if relative == "knowledge/alpha.md" else (self.root / relative).read_text(encoding="utf-8")
                destination.write_text(content, encoding="utf-8")
            with patch.object(rag_mvp, "ROOT", other_root):
                reordered = rag_mvp.load_documents()
            reordered_ids = {
                (chunk["source_path"], chunk["section_path"], chunk["text"]): chunk["chunk_id"]
                for doc in reordered for chunk in doc["chunks"]
            }
        self.assertEqual(original_chunks, reordered_ids)

    def test_search_rebuilds_stale_index_from_source_manifest(self) -> None:
        before_manifest = rag_mvp.source_manifest()
        (self.root / "knowledge" / "alpha.md").write_text(
            "---\ntopic: alpha\n---\n# Alpha\nnewterm updatedgamma.", encoding="utf-8"
        )

        results = rag_mvp.search("updatedgamma", index_path=self.index_path)
        after_index = json.loads(self.index_path.read_text(encoding="utf-8"))

        self.assertNotEqual(before_manifest, after_index["source_manifest"])
        self.assertTrue(any("updatedgamma" in r["text"] for r in results))
        self.assertEqual(rag_mvp.search("alphaterm", index_path=self.index_path), [])


if __name__ == "__main__":
    unittest.main()
