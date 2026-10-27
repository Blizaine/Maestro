"""CPU-only checks for small helpers in the staged MMGP 4.4 source."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"


class Mmgp43OffloadHelperTests(unittest.TestCase):
    def test_detect_main_towers_groups_module_list_blocks_in_isolated_process(self):
        script = r'''
import os
import sys

app = os.environ["MAESTRO_APP_ROOT"]
sys.path.insert(0, os.path.join(app, "mmgp_backends", "v43"))
sys.path.insert(1, app)
from mmgp import offload

torch = offload.torch
model = torch.nn.Module()
model.blocks = torch.nn.ModuleList(
    [torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)]
)
tower_names, tower_modules = offload._detect_main_towers(model, min_floors=2)
assert tower_names == ["blocks."], tower_names
assert tower_modules == list(model.blocks)
print("module-list tower detected")
'''
        environment = os.environ.copy()
        environment["MAESTRO_APP_ROOT"] = str(APP)
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [sys.executable, "-B", "-c", script],
            cwd=APP,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertIn("module-list tower detected", result.stdout)


if __name__ == "__main__":
    unittest.main()
