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

    def test_missing_runtime_markers_never_auto_start_an_update(self):
        for target, environment in (("sm_89", "app/env"), ("sm_120", "app/env-rtx50")):
            with self.subTest(target=target):
                menu = self.evaluate("""
const launcher=require('./pinokio.js');
const info={exists:name=>name===%s,running:()=>false,local:()=>null};
launcher.menu({gpu:'nvidia',gpu_target:%s,gpu_driver:'610.88',platform:'win32'},info)
  .then(v=>process.stdout.write(JSON.stringify(v)));
""" % (json.dumps(environment), json.dumps(target)))
                upgrade = next(item for item in menu if item.get("href") == "update.js")
                self.assertFalse(upgrade.get("default", False))
                self.assertFalse(any(item.get("default") for item in menu))
                if target == "sm_89":
                    self.assertTrue(any(item.get("href") == "start.js" for item in menu))

    def test_running_compatibility_app_keeps_web_ui_with_missing_runtime_markers(self):
        menu = self.evaluate("""
const launcher=require('./pinokio.js');
const info={exists:name=>name==='app/env',running:name=>name==='start.js',
  local:()=>({url:'http://127.0.0.1:49999',port:'49999'})};
launcher.menu({gpu:'nvidia',gpu_target:'sm_89',gpu_driver:'610.88',platform:'win32'},info)
  .then(v=>process.stdout.write(JSON.stringify(v)));
""")
        self.assertEqual(menu[0]["href"], "http://127.0.0.1:49999/?maestro_ui_policy=lan-v2")
        self.assertTrue(menu[0]["default"])
        self.assertFalse(any(item.get("href") == "update.js" for item in menu))

    def test_existing_update_still_shows_its_progress(self):
        menu = self.evaluate("""
const launcher=require('./pinokio.js');
const info={exists:name=>name==='app/env',running:name=>name==='update.js',local:()=>null};
launcher.menu({gpu:'nvidia',gpu_target:'sm_89',gpu_driver:'610.88',platform:'win32'},info)
  .then(v=>process.stdout.write(JSON.stringify(v)));
""")
        self.assertEqual(menu[0]["href"], "update.js")
        self.assertEqual(menu[0]["text"], "Updating")

    def test_menu_keeps_web_ui_default_and_passes_captured_backend(self):
        menu = self.evaluate("""
const launcher=require('./pinokio.js');
const info={exists:()=>true,running:n=>n==='start.js',local:()=>({url:'http://127.0.0.1:49999',port:'49999'})};
launcher.menu({gpu:'nvidia',gpu_target:'sm_86',platform:'win32'},info).then(v=>process.stdout.write(JSON.stringify(v)));
""")
        self.assertEqual(menu[0]["href"], "http://127.0.0.1:49999/?maestro_ui_policy=lan-v2")
        self.assertTrue(menu[0]["default"])
        actions = next(item for item in menu if item["text"] == "Developer tests")["menu"][:5]
        self.assertEqual(len(actions), 5)
        self.assertEqual(actions[0]["href"], "tests/pinokio/test_bench.js?mode=smoke")
        self.assertTrue(all(action["params"]["base_url"] == "http://127.0.0.1:49999" for action in actions))
        self.assertFalse(any(action.get("default") for action in actions))

    def test_menu_reports_the_nested_bench_uri_as_running(self):
        menu = self.evaluate("""
const launcher=require('./pinokio.js');
const info={exists:()=>true,
  running:n=>n==='start.js'||n==='tests/pinokio/test_bench.js',
  local:()=>({url:'http://127.0.0.1:49999',port:'49999'})};
launcher.menu({gpu:'nvidia',gpu_target:'sm_86',platform:'win32'},info).then(v=>process.stdout.write(JSON.stringify(v)));
""")
        bench = next(item for item in menu if item["text"] == "Developer tests (running)")
        self.assertEqual(bench["menu"][0]["href"], "tests/pinokio/test_bench.js?mode=smoke")
        self.assertEqual(bench["menu"][5]["href"], "tests/pinokio/test_bench_enable.js")

    def test_menu_still_reports_a_pre_migration_bench_as_running(self):
        menu = self.evaluate("""
const launcher=require('./pinokio.js');
const info={exists:()=>true,
  running:n=>n==='start.js'||n==='test_bench.js',
  local:()=>({url:'http://127.0.0.1:49999',port:'49999'})};
launcher.menu({gpu:'nvidia',gpu_target:'sm_86',platform:'win32'},info).then(v=>process.stdout.write(JSON.stringify(v)));
""")
        self.assertTrue(any(item["text"] == "Developer tests (running)" for item in menu))

    def test_3080_uses_compatibility_environment_and_structured_argv(self):
        launcher = self.evaluate("require('./tests/pinokio/test_bench.js')({gpu:'nvidia',gpu_target:'sm_86',platform:'win32'}).then(v=>process.stdout.write(JSON.stringify(v)))")
        step = launcher["run"][0]
        self.assertEqual(step["method"], "shell.run")
        self.assertEqual(step["params"]["venv"], "env")
        self.assertEqual(step["params"]["path"], "../../app")
        self.assertEqual(step["params"]["message"]["_"][:3], ["python", "-m", "testbench"])
        self.assertFalse(launcher.get("daemon", False))

    def test_4090_uses_same_marker_fallback_as_normal_start(self):
        launcher = self.evaluate("require('./tests/pinokio/test_bench.js')({gpu:'nvidia',gpu_target:'sm_89',gpu_driver:'610.88',platform:'win32'}).then(v=>process.stdout.write(JSON.stringify(v)))")
        environment = launcher["run"][0]["params"]["venv"]
        self.assertIn("../../app/env-sol/.maestro_sol_runtime_v1.installed", environment)
        self.assertIn("'env-sol' : 'env'", environment)

    def test_ssh_setup_checks_a_fresh_receipt_after_elevation(self):
        launcher = self.evaluate("require('./tests/pinokio/test_ssh_setup.js')({which:()=>null}).then(v=>process.stdout.write(JSON.stringify(v)))")
        elevated, verification = launcher["run"]
        self.assertTrue(elevated["params"]["sudo"])
        self.assertFalse(verification["params"].get("sudo", False))
        command = elevated["params"]["message"]["_"]
        self.assertEqual(elevated["params"]["path"], "../../app")
        self.assertIn("{{path.resolve(cwd, '../../app', 'scripts', 'setup_test_ssh.ps1')}}", command)
        self.assertEqual(verification["params"]["message"]["_"], command + ["-VerifyOnly"])
        self.assertIn("-RunId", command)

    def test_all_relocated_launcher_descriptors_resolve_the_profile_and_app_paths(self):
        descriptors = self.evaluate("""
const kernel = {gpu:'nvidia',gpu_target:'sm_89',gpu_driver:'610.88',platform:'win32',which:()=>null};
Promise.all([
  require('./tests/pinokio/test_bench_enable.js')(kernel),
  require('./tests/pinokio/test_ssh_client.js')(kernel),
  require('./tests/pinokio/test_ssh_setup.js')(kernel),
]).then(value => process.stdout.write(JSON.stringify(value)));
""")
        enable, client, setup = descriptors
        self.assertEqual(enable["run"][0]["params"]["path"], "../../app")
        self.assertIn("../../app/env-sol/.maestro_sol_runtime_v1.installed", enable["run"][0]["params"]["venv"])
        self.assertEqual(client["run"][0]["params"]["path"], "../../app")
        self.assertEqual(client["run"][0]["params"]["message"]["_"][1], "scripts/test_ssh_client.py")
        self.assertEqual(setup["run"][1]["params"]["path"], "../../app")
        self.assertTrue((ROOT / "scripts" / "pinokio" / "launcher_profile.js").is_file())
        self.assertFalse((ROOT / "launcher_profile.js").exists())
        self.assertTrue((ROOT / "docs" / "images" / "maestro-icon.png").is_file())
        self.assertFalse((ROOT / "maestro_simplified_icon_alpha.png").exists())

    def test_core_launcher_modules_load_after_the_profile_move(self):
        loaded = self.evaluate("""
const launchers = [
  'install.js', 'pinokio.js', 'start.js', 'start_classic.js',
  'start_sol.js', 'sol_install.js', 'torch.js', 'update.js',
];
for (const launcher of launchers) require('./' + launcher);
process.stdout.write(JSON.stringify(launchers));
""")
        self.assertEqual(len(loaded), 8)

    def test_existing_update_descriptor_still_pulls_before_using_the_root_torch_launcher(self):
        update = self.evaluate("""
require('./update.js')({gpu:'nvidia',gpu_target:'sm_89',gpu_driver:'610.88',platform:'win32'})
  .then(value => process.stdout.write(JSON.stringify(value)));
""")
        run = update["run"]
        self.assertEqual(run[0]["method"], "shell.run")
        self.assertEqual(run[0]["params"]["message"], "git pull")
        self.assertEqual(run[1]["method"], "jump")
        self.assertTrue(any(
            step.get("method") == "script.start" and step.get("params", {}).get("uri") == "torch.js"
            for step in run
        ))
        self.assertFalse(any(
            step.get("method") == "script.start" and step.get("params", {}).get("uri") == "scripts/pinokio/launcher_profile.js"
            for step in run
        ))


if __name__ == "__main__":
    unittest.main()
