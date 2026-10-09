"""Optional key-authenticated diagnostic client, run by the Pinokio desktop user."""
import argparse
import base64
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

APP = Path(__file__).resolve().parents[1]
KEY_DIR = APP / "settings" / "test-ssh"
OUTPUT = APP / "outputs" / "Test-Bench"


def execute(argv, timeout=30):
    return subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout, check=True)


def subprocess_failure_message(error):
    if isinstance(error, subprocess.CalledProcessError):
        summary = f"subprocess exited with status {error.returncode}"
    elif isinstance(error, subprocess.TimeoutExpired):
        summary = f"subprocess timed out after {error.timeout} seconds"
    else:
        summary = "subprocess failed"

    stderr = getattr(error, "stderr", None)
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    stderr = (stderr or "").strip()
    if stderr:
        if len(stderr) > 1500:
            stderr = "..." + stderr[-1500:]
        summary += "; stderr: " + stderr
    else:
        summary += " (no stderr output)"
    return "SSH diagnostics: " + summary


def prepare():
    if os.name != "nt":
        raise ValueError("This helper currently supports the Windows controller")
    keygen = shutil.which("ssh-keygen")
    if not keygen:
        raise ValueError("Install Windows OpenSSH Client first")
    KEY_DIR.mkdir(parents=True, exist_ok=True)
    key = KEY_DIR / "id_ed25519"
    if not key.exists():
        execute([keygen, "-t", "ed25519", "-N", "", "-f", str(key), "-C", "Maestro test bench"])
    # The script runs under the actual desktop account. Restrict this dedicated
    # key to its owner and SYSTEM; never print/read its private bytes.
    sid = execute(["whoami", "/user", "/fo", "csv", "/nh"]).stdout
    match = re.search(r"S-1-5-[0-9-]+", sid)
    if not match:
        raise ValueError("Could not resolve the key owner SID")
    execute(["icacls", str(key), "/inheritance:r", "/grant:r", "*" + match.group() + ":(F)", "*S-1-5-18:(F)"])
    public = key.with_suffix(".pub").read_text(encoding="ascii").strip()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    public_copy = OUTPUT / "test-controller-key.pub"
    public_copy.write_text(public + "\n", encoding="ascii")
    receipt = {"status": "prepared", "public_key_file": str(public_copy),
               "public_key": public, "fingerprint": execute([keygen, "-lf", str(public_copy)]).stdout.strip()}
    (OUTPUT / "ssh-client.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt), flush=True)


def diagnose():
    record = json.loads((APP / "settings" / "test_ssh_server.json").read_text(encoding="utf-8-sig"))
    host, account = record["host"], record["account_name"]
    if not re.fullmatch(r"[A-Za-z0-9.-]+", host) or not re.fullmatch(r"[A-Za-z0-9_.-]+", account):
        raise ValueError("Invalid SSH host/account")
    public = record["host_public_key"].split()
    if len(public) < 2 or public[0] != "ssh-ed25519" or not re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", public[1]):
        raise ValueError("Expected an Ed25519 host key from the trusted Pinokio setup receipt")
    KEY_DIR.mkdir(parents=True, exist_ok=True)
    known = KEY_DIR / "known_hosts"
    expected = host + " " + " ".join(public[:2]) + "\n"
    if known.exists() and known.read_text(encoding="ascii") != expected:
        raise ValueError("Server host key changed; inspect it rather than replacing the pinned key")
    known.write_text(expected, encoding="ascii")
    remote_root = str(record["remote_root"])
    if not re.fullmatch(r"[A-Za-z]:[\\/][A-Za-z0-9 _./\\-]+", remote_root) or ".." in remote_root:
        raise ValueError("Invalid remote Maestro root")
    script = r"""
$ErrorActionPreference = 'Stop'
$root = '__ROOT__'
$logs = @{}
foreach ($relative in @('logs\api\start.js\latest', 'logs\llm\llama-server.log')) {
  $target = Join-Path $root $relative
  if (Test-Path -LiteralPath $target) { $logs[$relative] = @(Get-Content -LiteralPath $target -Tail 250) }
}
$processes = @(Get-Process python,llama-server -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,WorkingSet64,Path)
$service = Get-Service sshd -ErrorAction SilentlyContinue | Select-Object Name,Status
@{user=(whoami);computer=$env:COMPUTERNAME;sshd=$service;processes=$processes;logs=$logs} | ConvertTo-Json -Depth 6 -Compress
""".replace("__ROOT__", remote_root)
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    ssh = shutil.which("ssh")
    if not ssh:
        raise ValueError("Windows OpenSSH Client was not found")
    command = [ssh, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=10",
               "-o", "UserKnownHostsFile=" + str(known), "-o", "IdentitiesOnly=yes", "-i", str(KEY_DIR / "id_ed25519"),
               account + "@" + host, "powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded]
    result = execute(command, timeout=30)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # Raw process paths and Maestro logs are private diagnostic evidence.
    (OUTPUT / "ssh-diagnostics.json").write_text(result.stdout, encoding="utf-8")
    parsed = json.loads(result.stdout)
    print(json.dumps({"status": "connected", "host": host, "user": parsed.get("user"),
                      "computer": parsed.get("computer"), "process_count": len(parsed.get("processes") or []),
                      "artifact": "ssh-diagnostics.json"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("prepare", "diagnose"), default="prepare")
    args = parser.parse_args()
    try:
        prepare() if args.action == "prepare" else diagnose()
        return 0
    except subprocess.SubprocessError as error:
        print(subprocess_failure_message(error), file=sys.stderr, flush=True)
        return 2
    except (OSError, ValueError, KeyError) as error:
        print("SSH diagnostics: " + str(error), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
