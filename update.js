const { runtimeProfile } = require("./launcher_profile")

module.exports = async (kernel) => {
  const runtime = runtimeProfile(kernel)
  const alreadyCurrentAndReady =
    `{{/already up[- ]to[- ]date/i.test(input.stdout) && exists('${runtime.marker}') ? 'uptodate' : 'build'}}`
  return {
    run: [{
    // Pull the latest launcher + app code (single monorepo, so this one
    // pull covers both `ui/` and `app/`). The NEXT step inspects this
    // pull's output: if the repo was already current, there is nothing
    // new to install or rebuild, so we skip straight to the end instead
    // of spending several minutes on a redundant dependency install +
    // UI build.
    method: "shell.run",
    params: {
      message: "git pull"
    }
  }, {
    // Branch on the git pull output (captured here as input.stdout — a
    // shell.run always returns its raw terminal content as stdout):
    //   - already current  -> jump to "uptodate" (log a notice, then end)
    //   - new commits found -> jump to "build"   (run the full update)
    // Matches both the modern "Already up to date" and the older git
    // "Already up-to-date" spelling, case-insensitively. If detection
    // ever fails (e.g. empty stdout), the regex simply won't match and
    // we fall through to "build" — the safe default is a full update,
    // never a wrongly-skipped rebuild.
    method: "jump",
    params: {
      // An already-current checkout still enters the build path when its
      // hardware-specific runtime marker is missing. This makes the RTX 50
      // CUDA 13 migration resumable after an interrupted Update.
      id: alreadyCurrentAndReady
    }
  }, {
    // Reached ONLY when the repo was already current (the "build" path
    // jumps over this step). Before halting, self-heal the seed-vc
    // component if it's missing (GPL-3.0, cloned from its own repo — see
    // install.js): a failed earlier clone shouldn't leave voice features
    // broken until the next code update.
    id: "uptodate",
    when: "{{!exists('app/postprocessing/seedvc/__init__.py')}}",
    method: "shell.run",
    params: {
      message: "git clone --depth 1 --branch v1.0.0 https://github.com/Blizaine/maestro-seedvc app/postprocessing/seedvc"
    }
  }, {
    method: "log",
    params: {
      raw: "Already up to date — no new commits pulled. Skipped dependency install and UI rebuild."
    },
    next: null
  }, {
    // Fetch the seed-vc component if missing (GPL-3.0, own repository —
    // see install.js). Runs at the top of the build path so the update
    // that removed the formerly-tracked tree clones it right back, and
    // any later update self-heals a failed clone. Keep the pinned tag in
    // sync with install.js.
    id: "build",
    when: "{{!exists('app/postprocessing/seedvc/__init__.py')}}",
    method: "shell.run",
    params: {
      message: "git clone --depth 1 --branch v1.0.0 https://github.com/Blizaine/maestro-seedvc app/postprocessing/seedvc"
    }
  }, {
    method: "shell.run",
    params: {
      venv: runtime.env,
      venv_python: runtime.python,
      path: "app",
      message: "uv pip install -r requirements.txt"
    }
  }, {
    // Skip torch.js when the marker file written by torch.js's last
    // successful run is still present — `torch + triton + sage + flash`
    // are already installed at the versions torch.js wants to install.
    // Saves ~60-120s + ~3 GB of redundant downloads on routine updates.
    //
    // Each hardware profile owns its marker. Bumping the marker in
    // launcher_profile.js makes this gate reinstall that profile on Update;
    // old markers remain harmless until Reset removes the environment.
    //
    // Recovery path: if torch ever ends up in a broken state (e.g. CPU
    // wheel installed where CUDA is expected) AND the marker is still
    // present, RTX 50 users can choose Advanced > Repair RTX 50 Runtime;
    // any user can delete their profile marker and re-run Update, or run
    // Reset for a clean slate.
    when: `{{!exists('${runtime.marker}')}}`,
    method: "script.start",
    params: {
      uri: "torch.js",
      params: {
        venv: runtime.env,
        path: "app",
        xformers: true
      }
    }
  }, {
    // Mirror of the install.js GGUF-kernels step — idempotent, so
    // re-runs cheaply on every update. Catches existing installs
    // up to the new behavior without forcing a reinstall.
    method: "shell.run",
    params: {
      venv: runtime.env,
      venv_python: runtime.python,
      path: "app",
      message: "python scripts/install_gguf_kernels.py"
    }
  }, {
    when: "{{exists('ui/package.json')}}",
    method: "shell.run",
    params: {
      path: "ui",
      message: [
        "npm install",
        "npm run build"
      ]
    }
  },
  // Update SAM 3.1 service (pull latest + reinstall) ONLY if SAM is
  // already installed. This way:
  //   - Users who never installed SAM (most users) don't get a slow
  //     conda env install they didn't ask for during a regular update.
  //   - Users who DID install SAM keep getting it kept up to date
  //     alongside the main app on every update.
  // Fresh-install path: install.js no longer runs sam_install.js;
  // users who want inpaint click "Install Inpaint Support" from the
  // Pinokio menu, which fires sam_install.js once. After that, this
  // gate is satisfied and SAM updates with every regular update.
  {
    when: "{{exists('app/services/sam/env')}}",
    method: "script.start",
    params: {
      uri: "sam_install.js"
    }
    }]
  }
}
