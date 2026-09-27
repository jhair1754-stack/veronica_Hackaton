from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import knowledge_traceability as trace


class TraceabilityTests(unittest.TestCase):
    def test_frontmatter_preserves_yaml_lists(self) -> None:
        metadata, body = trace.frontmatter(
            '---\ntitle: "Example"\ntarget_use:\n  - "one"\n  - "two"\n---\n# Body\n'
        )
        self.assertEqual(metadata["target_use"], ["one", "two"])
        self.assertIn("# Body", body)

    def test_markdown_heading_anchors_follow_github_slug_rules(self) -> None:
        self.assertEqual(trace.anchors("## Validación y evaluación\n## A/B\n"), {"validación-y-evaluación", "ab"})

    def test_audit_checks_local_paths_anchors_metadata_and_ignores_external_links(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "knowledge" / "eda").mkdir(parents=True)
            (root / "sources" / "book").mkdir(parents=True)
            source = root / "sources" / "book" / "chapter.md"
            source.write_text("# Chapter\n## Existing Section\nText\n", encoding="utf-8")
            (root / "knowledge" / "eda" / "audit.md").write_text(
                '---\ntitle: Audit\nknowledge_area: wrong\nstatus: draft\nversion: "1"\n'
                'created_from:\n  - "sources/book/"\ntarget_use:\n  - "tests"\n---\n'
                '[valid](../../sources/book/chapter.md#existing-section) '
                '[broken](../../sources/book/missing.md) '
                '[bad anchor](../../sources/book/chapter.md#missing) '
                '[web](https://example.com)\n',
                encoding="utf-8",
            )
            (root / "sources" / "source_catalog.md").write_text(
                "# Catalog\n| Chapter | File | Status |\n| --- | --- | --- |\n"
                "| 1 | [chapter](book/chapter.md) | `complete` |\n",
                encoding="utf-8",
            )
            previous_root = trace.ROOT
            try:
                trace.ROOT = root
                diagnostics = trace.audit()
            finally:
                trace.ROOT = previous_root

        errors = [item for item in diagnostics if item.error]
        self.assertTrue(any(item.kind == "ROTA" for item in errors))
        self.assertTrue(any(item.kind == "ANCLA" for item in errors))
        self.assertTrue(any("knowledge_area" in item.message for item in errors))
        self.assertFalse(any("example.com" in item.message for item in diagnostics))

    def test_registered_empty_playbooks_are_not_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "sources").mkdir()
            (root / "playbook").mkdir()
            files = trace.PENDING_PLAYBOOKS
            inventory = "\n".join(files)
            (root / "sources" / "README.md").write_text(inventory, encoding="utf-8")
            (root / "sources" / "source_catalog.md").write_text(inventory, encoding="utf-8")
            (root / "RAG_MVP.md").write_text(inventory, encoding="utf-8")
            for relative in files:
                (root / relative).write_bytes(b"")

            previous_root = trace.ROOT
            try:
                trace.ROOT = root
                diagnostics = trace.audit()
            finally:
                trace.ROOT = previous_root

        self.assertFalse([item for item in diagnostics if item.kind == "PLAYBOOK" and item.error])

    def test_explicit_prose_citations_are_checked_against_source_sections(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            knowledge_dir = root / "knowledge" / "metrics"
            source_dir = root / "sources" / "geron_hands_on_ml"
            knowledge_dir.mkdir(parents=True)
            source_dir.mkdir(parents=True)
            source_path = source_dir / "ch03_classification.md"
            source_path.write_text(
                '---\ntitle: Classification\ntopics:\n  - metrics\ntarget_knowledge_files: []\n---\n'
                "# Classification\n## 2.2 Validación\n## 8. Supuestos\n",
                encoding="utf-8",
            )
            knowledge_path = knowledge_dir / "metrics.md"
            knowledge_path.write_text(
                '---\ntitle: Metrics\nknowledge_area: metrics\nstatus: draft\nversion: "1"\n'
                'created_from:\n  - "sources/"\ntarget_use:\n  - "tests"\n---\n'
                "[Géron, cap. 3, §§2.2 y 8]\n",
                encoding="utf-8",
            )
            (root / "sources" / "source_catalog.md").write_text(
                "| C | File | Status |\n| --- | --- | --- |\n"
                "| 3 | [chapter](geron_hands_on_ml/ch03_classification.md) | `complete` |\n",
                encoding="utf-8",
            )

            previous_root = trace.ROOT
            try:
                trace.ROOT = root
                valid = trace.audit()
                knowledge_path.write_text(
                    knowledge_path.read_text(encoding="utf-8").replace("§§2.2 y 8", "§§2.2 y 9"),
                    encoding="utf-8",
                )
                invalid = trace.audit()
            finally:
                trace.ROOT = previous_root

        self.assertFalse([item for item in valid if item.kind == "SECCIÓN CITADA"])
        self.assertTrue([item for item in invalid if item.kind == "SECCIÓN CITADA"])


if __name__ == "__main__":
    unittest.main()
