"""CLI help must work without SCRATCH, checkpoints, or GPU initialization."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CliTests(unittest.TestCase):
    def test_help_from_another_directory(self):
        root = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment.pop("SCRATCH", None)
        environment.pop("SLURM_JOB_ID", None)
        with tempfile.TemporaryDirectory() as directory:
            for script in sorted((root / "scripts").glob("*.py")):
                with self.subTest(script=script.name):
                    result = subprocess.run(
                        [sys.executable, "-I", str(script), "--help"],
                        cwd=directory,
                        env=environment,
                        capture_output=True,
                        text=True,
                        timeout=30,
                        check=False,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("--project-root", result.stdout)
