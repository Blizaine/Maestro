import json
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("node"), "Node is required to evaluate Pinokio launcher routing")
class TestBenchLauncherTests(unittest.TestCase):
    def evaluate(self, code):
        result = subprocess.run([shutil.which("node"), "-e", code], cwd=ROOT, check=True,
                                capture_output=True, text=True, timeout=10)
        return json.loads(result.stdout)

    def test_menu_keeps_web_ui_default_and_passes_captured_backend(self):
        menu = self.evaluate("""
const launcher=require('./pinokio.js');
const info={exists:()=>true,running:n=>n==='start.js',local:()=>({url:'http://127.0.0.1:49999',port:'49999'})};
launcher.menu({gpu:'nvidia',gpu_target:'sm_86',platform:'win32'},info).then(v=>process.stdout.write(JSON.stringify(v)));
""")
        self.assertEqual(menu[0]["href"], "http://127.0.0.1:49999")
        self.assertTrue(menu[0]["default"])
        actions = next(item for item in menu if item["text"] == "Developer tests")["menu"][:5]
        self.assertEqual(len(actions), 5)
        self.assertTrue(all(action["params"]["base_url"] == menu[0]["href"] for action in actions))
        self.assertFalse(any(action.get("default") for action in actions))

    def test_3080_uses_compatibility_environment_and_structured_argv(self):
        launcher = self.evaluate("require('./test_bench.js')({gpu:'nvidia',gpu_target:'sm_86',platform:'win32'}).then(v=>process.stdout.write(JSON.stringify(v)))")
        step = launcher["run"][0]
        self.assertEqual(step["method"], "shell.run")
        self.assertEqual(step["params"]["venv"], "env")
        self.assertEqual(step["params"]["path"], "app")
        self.assertEqual(step["params"]["message"]["_"][:3], ["python", "-m", "testbench"])
        self.assertFalse(launcher.get("daemon", False))

    def test_4090_uses_same_marker_fallback_as_normal_start(self):
        launcher = self.evaluate("require('./test_bench.js')({gpu:'nvidia',gpu_target:'sm_89',gpu_driver:'610.88',platform:'win32'}).then(v=>process.stdout.write(JSON.stringify(v)))")
        environment = launcher["run"][0]["params"]["venv"]
        self.assertIn("app/env-sol/.maestro_sol_runtime_v1.installed", environment)
        self.assertIn("'env-sol' : 'env'", environment)

    def test_ssh_setup_checks_a_fresh_receipt_after_elevation(self):
        launcher = self.evaluate("require('./test_ssh_setup.js')({which:()=>null}).then(v=>process.stdout.write(JSON.stringify(v)))")
        elevated, verification = launcher["run"]
        self.assertTrue(elevated["params"]["sudo"])
        self.assertFalse(verification["params"].get("sudo", False))
        command = elevated["params"]["message"]["_"]
        self.assertIn("{{path.resolve(cwd, 'app', 'scripts', 'setup_test_ssh.ps1')}}", command)
        self.assertEqual(verification["params"]["message"]["_"], command + ["-VerifyOnly"])
        self.assertIn("-RunId", command)


if __name__ == "__main__":
    unittest.main()
