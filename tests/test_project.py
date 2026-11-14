"""Configuration and metadata tests requiring neither a GPU nor network."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ltr_system.project import (
    encoder_config,
    project_root,
    read_json,
    validate_revision,
    write_json,
)


class ProjectTests(unittest.TestCase):
    def test_explicit_root_wins_over_scratch(self):
        with patch.dict(os.environ, {"LTR_PROJECT_ROOT": "custom", "SCRATCH": "scratch"}):
            self.assertEqual(project_root(), Path("custom").resolve())

    def test_scratch_default(self):
        with patch.dict(os.environ, {"SCRATCH": "scratch"}, clear=True):
            self.assertEqual(project_root(), (Path("scratch") / "learning-to-rank").resolve())

    def test_config_pins_immutable_revision(self):
        config = encoder_config()
        self.assertEqual(len(validate_revision(config["revision"])), 40)
        self.assertEqual(config["dimension"], 384)

    def test_rejects_moving_or_invalid_revisions(self):
        for revision in ["main", "abc1234", "z" * 40, None]:
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                validate_revision(revision)

    def test_json_replacement_and_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested/manifest.json"
            write_json(path, {"version": 1})
            write_json(path, {"version": 2})
            self.assertEqual(read_json(path), {"version": 2})
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_failed_write_preserves_previous_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_json(path, {"version": 1})
            with self.assertRaises(TypeError):
                write_json(path, {"bad": object()})
            self.assertEqual(read_json(path), {"version": 1})
            self.assertEqual(list(path.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
