"""Guard the environment-capture call sites in launch.py.

launch.py is over 31,000 lines and tracks an upstream that changes it
often. The capture itself is five one-line additions inside sidecar dicts,
which is exactly the kind of thing a merge can drop silently: generation
keeps working, nothing errors, and the sidecars quietly stop recording how
they were produced. That loss is only noticeable weeks later, by which
point the runs are gone.

So this asserts the wiring rather than the behaviour. If an upstream merge
removes a call site, this fails instead of the data disappearing.
"""

import ast
import re
import sys
import unittest
from pathlib import Path

APP = Path(__file__).parents[1] / "app"
LAUNCH = APP / "launch.py"

# Every function that writes a .meta.json for something the user generated.
EXPECTED_CALL_SITES = {
    "_run_generation",                      # ordinary video / image / audio
    "_write_tool_sidecar",                  # face refiner, upscale, media flow
    "_write_recast_shot_aware_sidecar",
    "_write_repaint_shot_aware_sidecar",
    "_write_outpaint_shot_aware_sidecar",
}


class EnvSnapshotWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = LAUNCH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_helper_still_exists(self):
        names = {
            node.name
            for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef)
        }
        self.assertIn(
            "_env_snapshot", names,
            "_env_snapshot() has gone from launch.py; sidecars will no longer "
            "record the settings a run was produced under.",
        )

    def test_every_generation_path_captures_the_environment(self):
        """Each expected writer must still call _env_snapshot()."""
        found = set()
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name not in EXPECTED_CALL_SITES:
                continue
            calls = {
                sub.func.id
                for sub in ast.walk(node)
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
            }
            if "_env_snapshot" in calls:
                found.add(node.name)

        missing = EXPECTED_CALL_SITES - found
        self.assertEqual(
            missing, set(),
            f"These sidecar writers no longer capture the environment: "
            f"{sorted(missing)}. Re-add \"environment\": _env_snapshot() to "
            f"each one's sidecar dict.",
        )

    def test_call_site_count_has_not_drifted(self):
        """A bare count, so a site added outside a known writer is noticed."""
        count = len(re.findall(r'"environment":\s*_env_snapshot\(\)', self.source))
        self.assertEqual(
            count, len(EXPECTED_CALL_SITES),
            f"Found {count} environment capture sites, expected "
            f"{len(EXPECTED_CALL_SITES)}. If a new generation path was added, "
            f"add it to EXPECTED_CALL_SITES; if one was removed, restore it.",
        )

    def test_snapshot_module_is_present_and_importable(self):
        sys.path.insert(0, str(APP))
        try:
            from services.env_snapshot import build_env_snapshot
        finally:
            sys.path.remove(str(APP))
        snapshot = build_env_snapshot(
            {"vram_allocator": "vmm", "services": {"civitai_api_key": "secret"}},
            "3.2.0",
        )
        self.assertEqual(snapshot["vram_allocator"], "vmm")
        self.assertNotIn("secret", str(snapshot))


if __name__ == "__main__":
    unittest.main()
