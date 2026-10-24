"""Run the native result/command blocks without loading Maestro or a GPU."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def block(path, condition):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test) == condition)
    return compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), path, "exec")


class SpeechResultWiringTests(unittest.TestCase):
    def test_native_partial_report_is_attached_and_warnings_are_emitted_only_for_fish(self):
        code = block("app/wgp.py", "model_def.get('architecture') == 'fish_drama3'")
        report = {"incomplete": True, "completed_chunks": 1, "total_chunks": 2}
        for architecture in ("fish_drama3", "other_tts"):
            with self.subTest(architecture=architecture):
                commands = []
                ns = {"model_def": {"architecture": architecture}, "artifact_metadata": {"sample_rate": 44100},
                      "samples": {"speech_generation": report, "generation_warnings": ["Part 2 reached its limit", "", None]},
                      "send_cmd": lambda *args: commands.append(args)}
                exec(code, ns)
                self.assertEqual(ns["artifact_metadata"]["sample_rate"], 44100)
                if architecture == "fish_drama3":
                    self.assertEqual(ns["artifact_metadata"]["speech_generation"], report)
                    self.assertEqual(commands, [("generation_warning", "Part 2 reached its limit")])
                else:
                    self.assertNotIn("speech_generation", ns["artifact_metadata"])
                    self.assertEqual(commands, [])

    def test_generation_warning_is_deduplicated_and_persisted_without_failing_job(self):
        code = block("app/launch.py", "cmd == 'generation_warning'")
        job = {"status": "running", "finishing_warnings": ["Existing warning"]}
        changes = []

        def update_job(target, **fields):
            changes.append(fields)
            target.update(fields)
            return True

        ns = {"cmd": "generation_warning", "job": job, "update_job": update_job, "print": lambda *args: None}
        for warning in ("Part 2 incomplete", "Part 2 incomplete", "", None):
            exec(code, {**ns, "data": warning})
        self.assertEqual(job["status"], "running")
        self.assertEqual(job["finishing_warnings"], ["Existing warning", "Part 2 incomplete"])
        self.assertEqual(len(changes), 1)

    def test_classic_metadata_retains_the_exact_report(self):
        tree = ast.parse((ROOT / "app/wgp.py").read_text(encoding="utf-8"))
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                    and ast.unparse(n.test) == 'isinstance(speech_generation, dict)'
                    and any(isinstance(child, ast.Assign) and 'configs' in ast.unparse(child) for child in n.body))
        report = {"incomplete": True, "warnings": ["Stopped before the last part"]}
        ns = {"speech_generation": report, "configs": {"duration_seconds": 0}}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "wgp.py", "exec"), ns)
        self.assertEqual(ns["configs"]["speech_generation"], report)


if __name__ == "__main__":
    unittest.main()
