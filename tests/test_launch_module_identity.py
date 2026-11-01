"""Native-script and imported callbacks must share one live launch module."""
import ast
from pathlib import Path
import subprocess
import sys


LAUNCH = Path(__file__).resolve().parents[1] / "app" / "launch.py"


def bootstrap_identity_source():
    tree = ast.parse(LAUNCH.read_text(encoding="utf-8"))
    # Execute actual entry-point statements before application bootstrap,
    # without importing Torch or constructing a server in the CPU suite.
    statements = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_app_dir"
            for target in node.targets
        ):
            break
        if isinstance(node, ast.Import) and any(alias.name == "sys" for alias in node.names):
            statements.append(node)
        if isinstance(node, ast.If) and "__name__" in ast.unparse(node.test):
            statements.append(node)
    return ast.unparse(ast.Module(body=statements, type_ignores=[]))


def test_native_script_director_import_reuses_live_backend_module(tmp_path):
    source = bootstrap_identity_source()
    script = tmp_path / "launch.py"
    script.write_text(source + '''
import importlib
assert __name__ == "__main__", "callback re-executed server bootstrap"
backend_identity = object()
callback_module = importlib.import_module("launch")
assert callback_module is sys.modules["__main__"]
assert callback_module.backend_identity is backend_identity
''', encoding="utf-8")
    subprocess.run([sys.executable, str(script)], check=True, capture_output=True, timeout=15)


def test_imported_launch_does_not_replace_another_main_module():
    import types

    main = sys.modules["__main__"]
    module = types.ModuleType("launch")
    exec(bootstrap_identity_source(), module.__dict__)
    assert sys.modules["__main__"] is main
