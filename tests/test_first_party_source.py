"""Syntax guard for Python modules that ordinary unit tests do not import."""

from __future__ import annotations

import os
import unittest


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SOURCE_ROOTS = ("app", "scripts", "tests")
_SKIP_DIRS = {
    ".git",
    "__pycache__",
    "ckpts",
    "loras",
    "outputs",
    "temp",
    "uploads",
}


class TestFirstPartyPythonSource(unittest.TestCase):
    def test_every_first_party_python_file_compiles(self):
        checked = 0
        for relative_root in _SOURCE_ROOTS:
            source_root = os.path.join(_ROOT, relative_root)
            for directory, dirs, filenames in os.walk(source_root):
                dirs[:] = [
                    name for name in dirs
                    if name not in _SKIP_DIRS and not name.startswith("env")
                ]
                for filename in filenames:
                    if not filename.endswith(".py"):
                        continue
                    path = os.path.join(directory, filename)
                    with self.subTest(path=os.path.relpath(path, _ROOT)):
                        with open(path, "r", encoding="utf-8-sig") as handle:
                            compile(handle.read(), path, "exec")
                    checked += 1
        self.assertGreater(checked, 1000)


if __name__ == "__main__":
    unittest.main()
