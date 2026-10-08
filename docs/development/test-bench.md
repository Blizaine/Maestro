# Maestro test bench

The test bench runs the same bounded cases in each PC's normal Pinokio
runtime. Pinokio controls the test action; the worker talks only to that PC's
loopback Maestro API. The optional controller collects reports over the
existing private Pinokio app connection. No new public server is introduced.

## Manual workflow

1. Update both test installations to the intended Dev revision and start
   Maestro normally in Pinokio. Leave both generation queues idle.
2. On either Pinokio Maestro page, open **Developer tests → Diagnostics only**.
   This reads runtime, hardware, jobs and bounded logs without generating.
3. **Prompt regression tests** uses the opt-in prompt bench. Enable it once
   with **Developer tests → Enable local prompt tests**, or
   `python -m promptbench enable` from the managed `app` environment,
   then restart Maestro through Pinokio while idle. Use installed writers;
   missing assets are reported as untested rather than downloaded.
4. **H3 cold / warm baseline** runs a fixed short silent kite clip twice.
   **Memory profile comparison** tests a small hardware-appropriate matrix.
   The worker reuses the memory benchmark's settings backup and restoration.
5. **Bounded full test suite** combines diagnostics, prompt tests and the H3
   baseline. It is a manual action; installing the harness does not schedule it.

Do not stop a running launcher to cancel an in-flight inference. A `STOP` file
in the run directory prevents new cases. An uncertain receipt or stale lock
requires checking jobs, logs and the owning process before another run.
The harness never cancels another person's job. If a benchmark settings
restore needs recovery, use its saved `settings-backup.json` and the memory
benchmark's `--restore-only` command while idle.

Reports and bundles are private artifacts under `app/outputs/Test-Bench`.
The ZIP contains test requests, results, metadata and bounded allowlisted
logs; it excludes model weights, user configuration and generated videos.
Videos remain in their dedicated test workspace. Log excerpts can contain
personal prompts or paths, so inspect them before sharing externally.

## Fleet controller

Copy `app/scripts/test_fleet.example.json` to
`app/settings/test_fleet.json` and replace the two refs with the canonical
refs returned by `pterm search Maestro`. This ignored configuration stays on
the controller PC. Do not substitute the second PC's installation path as a
local path, and do not hardcode a backend port: the controller discovers it
on every run.

Run in a Pinokio-managed terminal from the repository root:

```powershell
python app/scripts/test_fleet.py --mode smoke
python app/scripts/test_fleet.py --mode prompt
python app/scripts/test_fleet.py --mode render
python app/scripts/test_fleet.py --mode compare
python app/scripts/test_fleet.py --mode nightly
```

`--mode nightly` means the bounded full suite; it does not install a schedule.
If `pterm` is not inherited by a separate shell, supply `--pterm` with the
path returned by `pterm which pterm`. Supply repeated `--ref` options to
override the saved targets. `--collect-only` fetches existing evidence
without launching a test. Each invocation gets a new private output directory
under `.codex-tmp/test-fleet`; `--output` can select another fresh directory.

The controller skips offline/busy machines. It never wakes, updates, starts,
stops or restarts Maestro, and never retries an ambiguous dispatch. Persistent
receipts in `app/settings/test-fleet-state` protect later invocations after a
transport failure. After inspecting a matching terminal result, use
`--collect-only` to reconcile it. A controller deadline leaves remote work
intact. Each machine is visited serially; a STOP file at the controller output
root prevents dispatching the next machine.

## Reading artifacts through the API

The receipt is available through the existing file endpoint. Use the API URL
reported by Pinokio for the current machine; remote callers use its external
ready URL. The receipt gives safe basename-only ZIP/report names.

```javascript
const receipt = await fetch(`${baseUrl}/api/v1/file/testbench-latest.json?workspace=Test-Bench`).then(r => r.json());
```

```python
import json, urllib.request
receipt = json.load(urllib.request.urlopen(base_url + '/api/v1/file/testbench-latest.json?workspace=Test-Bench'))
```

```sh
curl "$MAESTRO_URL/api/v1/file/testbench-latest.json?workspace=Test-Bench"
```

There is no remote prompt-bench API. Its developer endpoint remains loopback
only; the test worker runs locally on the source machine.

## Comparing results

Keep the source revision, case/suite digest, exact checkpoint identity,
seed, geometry and content fixed for comparisons. Record runtime, GPU/driver,
RAM/VRAM, quantization, attention, profile, preview and effective memory settings.
Compare cold and warm runs separately. Performance baselines belong to each
machine/workload/runtime; a 12 GB GPU with 32 GB RAM needs different offloading
than a 24 GB GPU with 128 GB RAM. Cross-machine timings are descriptive and
must not be treated as a regression threshold.

The initial memory comparison takes one cold sample per profile and changes
only the video memory profile. It is an exploratory check, not enough evidence
to select an Auto-tune winner. Use repeated cold and warm runs before promoting
a performance setting.

Prompt reports preserve repair attempts, warnings and native compilation
fallbacks. A completed enhancement is not necessarily warning-free. Mechanical
checks do not prove creative quality, lip sync, identity retention or motion;
review the prompts and rendered videos separately.

## Optional Windows SSH diagnostics

SSH is for deeper filesystem/process diagnostics. Normal test collection works
without it. Windows OpenSSH setup needs one elevated session on the test PC;
the provided `test_ssh_setup.js` invokes the fixed
`app/scripts/setup_test_ssh.ps1` helper through Pinokio's administrative shell.
Windows may require the person at that PC to approve UAC.
The action verifies a fresh setup receipt after the elevated shell returns.
If no receipt appears, check the Windows administrator prompt. A receipt with
`status: running` means setup reached the recorded phase; wait before retrying.
A failed receipt records the phase and Windows error.

Prepare a dedicated Ed25519 key under the actual Pinokio desktop account:

```powershell
pterm start test_ssh_client.js --ref YOUR_CONTROLLER_REF -- --action=prepare
```

Its **private** key stays in ignored `app/settings/test-ssh`, restricted to
that Windows account and SYSTEM. The action writes a public-key receipt to
`ssh-client.json` in the Test-Bench workspace. Upload only its `.pub` file with `pterm upload REF FILE`.
Use the returned source-machine path as `public_key_file`. Then run:

```powershell
pterm start test_ssh_setup.js --ref YOUR_TEST_PC_REF -- --public_key_file=RETURNED_PUBLIC_KEY_PATH --client_address=CONTROLLER_PRIVATE_IP --account_name=EXISTING_LOCAL_WINDOWS_USER
```

The helper installs the Windows OpenSSH capability if needed, appends the
public key for the selected existing local account, applies Windows authorized
key ACLs, starts sshd, and creates a Private/Domain firewall rule limited to
that one controller IP. On a new install it disables Windows' newly created
broad default SSH rule. It preserves pre-existing SSH/firewall configuration;
review existing rules separately if the service was already installed.
It does not change password authentication policy, create users, forward router
ports, or copy private keys to the server.

The resulting `ssh-setup.json` in the Test-Bench workspace includes the server's
public host key. Verify/pin that key from the trusted Pinokio setup result before
connecting with `ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -i KEY USER@HOST`.
Do not bypass host-key validation. Local or domain Windows accounts can use
OpenSSH keys; Microsoft Entra-only accounts need a suitable local account.
See Microsoft's [installation guide](https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse)
and [key/ACL guide](https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_keymanagement).

For a bounded diagnostic check, save the setup receipt in ignored
`app/settings/test_ssh_server.json` on the controller and add `host` (the
source PC's reachable private address) and `remote_root` (its Maestro path).
Then run `pterm start test_ssh_client.js --ref YOUR_CONTROLLER_REF -- --action=diagnose`.
This verifies the pinned server key and collects only the selected process
metadata and last 250 lines of Maestro/LLM logs. The result is
`ssh-diagnostics.json` in the controller's Test-Bench workspace. A changed
host key is rejected for manual review.

SSH sessions should inspect diagnostics and run bounded test commands.
Continue using the normal Pinokio Start/Stop/Update actions for Maestro's
lifecycle so environment selection and launcher logs remain consistent.
