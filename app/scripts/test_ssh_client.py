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
PTY_JSON_BEGIN = "MAESTRO_SSH_JSON_BEGIN"
PTY_JSON_END = "MAESTRO_SSH_JSON_END"
PTY_JSON_CHUNK_SIZE = 64
PTY_JSON_MAX_BYTES = 8 * 1024 * 1024


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


def decode_pty_json(output):
    """Decode the framed, ASCII-safe payload emitted through a remote PTY."""
    # ConPTY/OpenSSH can decorate terminal output. Remove OSC before CSI so an
    # OSC payload containing CSI-looking text cannot affect frame detection.
    output = re.sub(r"\x1b\].*?(?:\x07|\x1b\\|$)", "", output, flags=re.DOTALL)
    output = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", output)
    lines = output.splitlines()
    begin = [index for index, line in enumerate(lines) if line == PTY_JSON_BEGIN]
    end = [index for index, line in enumerate(lines) if line == PTY_JSON_END]
    if len(begin) != 1 or len(end) != 1 or begin[0] >= end[0]:
        raise ValueError("PTY diagnostics transport is missing or has ambiguous JSON frame markers")

    chunks = lines[begin[0] + 1:end[0]]
    if not chunks or any(not chunk or len(chunk) > PTY_JSON_CHUNK_SIZE for chunk in chunks):
        raise ValueError("PTY diagnostics transport contains an empty or oversized JSON chunk")
    encoded = "".join(chunks)
    if len(encoded) > ((PTY_JSON_MAX_BYTES + 2) // 3) * 4:
        raise ValueError("PTY diagnostics JSON frame exceeds the size limit")
    try:
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > PTY_JSON_MAX_BYTES or base64.b64encode(raw).decode("ascii") != encoded:
            raise ValueError("non-canonical or oversized base64")
        decoded = raw.decode("utf-8", errors="strict")
        payload = json.loads(decoded, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("PTY diagnostics JSON frame is malformed") from error
    if not isinstance(payload, dict):
        raise ValueError("PTY diagnostics JSON frame must contain an object")
    return payload, decoded


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


def diagnose(server_config=None):
    record_path = Path(server_config) if server_config is not None else APP / "settings" / "test_ssh_server.json"
    record = json.loads(record_path.read_text(encoding="utf-8-sig"))
    host, account = record["host"], record["account_name"]
    request_pty = record.get("request_pty", False)
    if not isinstance(request_pty, bool):
        raise ValueError("Invalid SSH server configuration: request_pty must be a boolean")
    if not re.fullmatch(r"[A-Za-z0-9.-]+", host) or not re.fullmatch(r"[A-Za-z0-9_.-]+(?:\\[A-Za-z0-9_.-]+)?", account):
        raise ValueError("Invalid SSH host/account")
    public = record["host_public_key"].split()
    if len(public) < 2 or public[0] != "ssh-ed25519" or not re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", public[1]):
        raise ValueError("Expected an Ed25519 host key from the trusted Pinokio setup receipt")
    KEY_DIR.mkdir(parents=True, exist_ok=True)
    # Additional machines get a separate pinned host file. Keep the original
    # diagnostics configuration and its host-key pin intact.
    known = KEY_DIR / ("known_hosts" if server_config is None else "known_hosts-" + host)
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
  if (Test-Path -LiteralPath $target) { $logs[$relative] = @(Get-Content -LiteralPath $target -Tail 250 | ForEach-Object { $_.ToString() }) }
}
$processes = @(Get-Process python,llama-server -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,WorkingSet64,Path)
$service = Get-Service sshd -ErrorAction SilentlyContinue | Select-Object Name,Status
""".replace("__ROOT__", remote_root)
    payload_line = "@{user=(whoami);computer=$env:COMPUTERNAME;sshd=$service;processes=$processes;logs=$logs}"
    if request_pty:
        script += f"""
$json = {payload_line} | ConvertTo-Json -Depth 6 -Compress
$encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($json))
[Console]::Out.WriteLine('{PTY_JSON_BEGIN}')
for ($offset = 0; $offset -lt $encoded.Length; $offset += {PTY_JSON_CHUNK_SIZE}) {{
  $count = [Math]::Min({PTY_JSON_CHUNK_SIZE}, $encoded.Length - $offset)
  [Console]::Out.WriteLine($encoded.Substring($offset, $count))
}}
[Console]::Out.WriteLine('{PTY_JSON_END}')
"""
    else:
        # Preserve the legacy no-PTY stdout shape exactly for existing machines.
        script += payload_line + " | ConvertTo-Json -Depth 6 -Compress\n"
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    ssh = shutil.which("ssh")
    if not ssh:
        raise ValueError("Windows OpenSSH Client was not found")
    command = [ssh, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=10",
               "-o", "UserKnownHostsFile=" + str(known), "-o", "IdentitiesOnly=yes", "-i", str(KEY_DIR / "id_ed25519")]
    if request_pty:
        command.append("-tt")
    command.extend([account + "@" + host, "powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded])
    result = execute(command, timeout=30)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # Raw process paths and Maestro logs are private diagnostic evidence.
    artifact = "ssh-diagnostics.json" if server_config is None else "ssh-diagnostics-" + host + ".json"
    if request_pty:
        parsed, artifact_text = decode_pty_json(result.stdout)
    else:
        artifact_text = result.stdout
        parsed = json.loads(result.stdout)
    (OUTPUT / artifact).write_text(json.dumps(parsed, indent=2) if request_pty else artifact_text,
                                   encoding="utf-8")
    print(json.dumps({"status": "connected", "host": host, "user": parsed.get("user"),
                      "computer": parsed.get("computer"), "process_count": len(parsed.get("processes") or []),
                      "artifact": artifact}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("prepare", "diagnose"), default="prepare")
    parser.add_argument("--server-config", type=Path, help="Private setup receipt for another SSH test machine")
    args = parser.parse_args()
    try:
        if args.action == "prepare":
            prepare()
        elif args.server_config is None:
            diagnose()
        else:
            diagnose(args.server_config)
        return 0
    except subprocess.SubprocessError as error:
        print(subprocess_failure_message(error), file=sys.stderr, flush=True)
        return 2
    except (OSError, ValueError, KeyError) as error:
        print("SSH diagnostics: " + str(error), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
