# Contributing to Maestro

Thanks for your interest in improving Maestro! This is a local-first AI
video/image/music studio built on the [Wan2GP](https://github.com/deepbeepmeep/Wan2GP)
pipeline and distributed through [Pinokio](https://pinokio.computer).

## Getting set up

Maestro is a Pinokio app, so the easiest dev loop is:

1. Install Maestro through Pinokio (see the [README](README.md)). This creates
   the Python environment in `app/env/` and installs the app.
2. Edit the source in place. The layout:
   - **Launcher scripts** (`install.js`, `start.js`, `update.js`, `reset.js`,
     `pinokio.js`) live at the repo root.
   - **Backend** — `app/`: FastAPI endpoints in `app/launch.py`, the generation
     pipeline in `app/wgp.py`, and services (LLM, Director, recipes, etc.) in
     `app/services/`.
   - **Frontend** — `ui/`: a React + TypeScript + Tailwind app; global state in
     `ui/src/stores/useStore.ts`.
3. After changing the UI, rebuild it:
   ```
   cd ui
   npm install
   npm run build
   ```
   Pinokio's **Update** flow does this automatically; during active dev you can
   run it yourself.

## Before you open a PR

CI runs three checks on every PR — please run them locally first:

```bash
# 1. Clean-repo guard (see below) — must pass
python scripts/verify_clean_repo.py

# 2. Python syntax on the modules you touched
python -m compileall -q app/services app/launch.py scripts

# 3. UI type-check + build
cd ui && npm run build
```

### The clean-repo guard

`scripts/verify_clean_repo.py` enforces that certain **locally-generated or
machine-specific artifacts never get committed** — downloaded weights, CivitAI
metadata sidecars, per-LoRA generated guides, and per-checkpoint finetune JSONs.
These are all gitignored by design; the guard is the backstop that keeps them
out of the published tree. If it fails, it prints exactly what leaked and where.
Don't work around it — fix the leak (usually a file that should be gitignored
got `git add`-ed).

## Conventions

- **Match the surrounding code.** Follow the naming, structure, and comment
  style already in the file you're editing.
- **Keep the app local-first.** No telemetry, no phone-home, no required
  accounts. External API providers (OpenAI/Anthropic/etc.) stay strictly
  opt-in and off by default.
- **Third-party components keep their own licenses.** Notably the GPL-3.0
  seed-vc voice component is fetched from its own repository at install time
  (see the README license section) rather than vendored here — don't commit it
  back into `app/postprocessing/seedvc/`.

## Reporting bugs

Please use the **Bug report** issue template — it asks for your logs
(`logs/api/latest` in the Pinokio app folder) and GPU/VRAM/OS, which is almost
always what's needed to reproduce a local-generation issue.

## License

Maestro is released under the WanGP Non-Commercial Evaluation License (inherited
from upstream Wan2GP). By contributing you agree your contributions are licensed
under the same terms. See [LICENSE](LICENSE).
