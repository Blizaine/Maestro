module.exports = {
  run: [{
    // Pull the latest launcher + app code (single monorepo, so this one
    // pull covers both `ui/` and `app/`). The next step inspects this
    // output: an unchanged Maestro skips its main dependency/UI rebuild,
    // but still reaches the bundled-runtime maintenance section.
    method: "shell.run",
    params: {
      message: "git pull"
    }
  }, {
    // Branch on the git pull output (captured here as input.stdout — a
    // shell.run always returns its raw terminal content as stdout):
    //   - already current and complete -> jump to "uptodate" and stop
    //   - old install missing native 3D -> jump to "build" for migration
    //   - new commits found             -> jump to "build" for the full update
    // Matches both the modern "Already up to date" and the older git
    // "Already up-to-date" spelling, case-insensitively. If detection
    // ever fails (e.g. empty stdout), the regex simply won't match and
    // we fall through to "build" — the safe default is a full update,
    // never a wrongly-skipped rebuild.
    method: "jump",
    params: {
      id: "{{/already up[- ]to[- ]date/i.test(input.stdout) && exists('app/services/hunyuan3d/env/.maestro_hunyuan3d_v1.installed') && exists('app/services/hunyuan3d/vendor/Hunyuan3D-2') && exists('app/services/hunyuan3d/vendor/Hunyuan3D-2.1') && exists('app/postprocessing/seedvc/__init__.py') ? 'uptodate' : 'build'}}"
    }
  }, {
    id: "uptodate",
    method: "log",
    params: {
      raw: "Already up to date. Maestro and its bundled runtimes are installed."
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
      venv: "env",
      path: "app",
      message: "uv pip install -r requirements.txt"
    }
  }, {
    // Skip torch.js when the marker file written by torch.js's last
    // successful run is still present — `torch + triton + sage + flash`
    // are already installed at the versions torch.js wants to install.
    // Saves ~60-120s + ~3 GB of redundant downloads on routine updates.
    //
    // When bumping ANY of those package versions inside torch.js, ALSO
    // bump the `_v1` suffix here AND in torch.js's fs.write step. The
    // old marker becomes stale, this `!exists(new_marker)` gate evaluates
    // true on the next update, torch.js runs, and the new marker is
    // written. Old marker stays as harmless cruft until reset.js (which
    // wipes app/env entirely).
    //
    // Recovery path: if torch ever ends up in a broken state (e.g. CPU
    // wheel installed where CUDA is expected) AND the marker is still
    // present, the user can manually delete
    // `app/env/.maestro_torch_v1.installed` and re-run Update to force
    // a full reinstall — or run Reset for a clean slate.
    when: "{{!exists('app/env/.maestro_torch_v1.installed')}}",
    method: "script.start",
    params: {
      uri: "torch.js",
      params: {
        venv: "env",
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
      venv: "env",
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
  },
  // Existing installations that predate native 3D reach this section through
  // the normal full update path. Model weights remain lazy downloads.
  {
    when: "{{!exists('app/postprocessing/seedvc/__init__.py')}}",
    method: "shell.run",
    params: {
      message: "git clone --depth 1 --branch v1.0.0 https://github.com/Blizaine/maestro-seedvc app/postprocessing/seedvc"
    }
  }, {
    when: "{{!exists('app/services/hunyuan3d/vendor/Hunyuan3D-2')}}",
    method: "shell.run",
    params: {
      message: "git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2 app/services/hunyuan3d/vendor/Hunyuan3D-2"
    }
  }, {
    when: "{{exists('app/services/hunyuan3d/vendor/Hunyuan3D-2/.git')}}",
    method: "shell.run",
    params: {
      path: "app/services/hunyuan3d/vendor/Hunyuan3D-2",
      message: "git pull --ff-only"
    }
  }, {
    when: "{{!exists('app/services/hunyuan3d/vendor/Hunyuan3D-2.1')}}",
    method: "shell.run",
    params: {
      message: "git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1 app/services/hunyuan3d/vendor/Hunyuan3D-2.1"
    }
  }, {
    when: "{{exists('app/services/hunyuan3d/vendor/Hunyuan3D-2.1/.git')}}",
    method: "shell.run",
    params: {
      path: "app/services/hunyuan3d/vendor/Hunyuan3D-2.1",
      message: "git pull --ff-only"
    }
  }, {
    method: "shell.run",
    params: {
      conda: { path: "app/services/hunyuan3d/env", python: "3.10" },
      message: [
        "uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128",
        "uv pip install -r app/services/hunyuan3d/requirements.txt"
      ]
    }
  }, {
    // Remove the accidental nested conda environment produced by the initial
    // native-3D migration before compiling into Maestro's real 3D runtime.
    when: "{{exists('app/services/hunyuan3d/vendor/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer/app')}}",
    method: "fs.rm",
    params: {
      path: "app/services/hunyuan3d/vendor/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer/app"
    }
  }, {
    method: "shell.run",
    params: {
      conda: { path: "../../../../../env", python: "3.10" },
      path: "app/services/hunyuan3d/vendor/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer",
      message: "uv pip install --no-build-isolation -e ."
    }
  }, {
    method: "shell.run",
    params: {
      conda: { path: "../../../../../env", python: "3.10" },
      path: "app/services/hunyuan3d/vendor/Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer",
      message: "uv pip install --no-build-isolation -e ."
    }
  }, {
    method: "shell.run",
    params: {
      conda: { path: "../../../../env", python: "3.10" },
      path: "app/services/hunyuan3d/vendor/Hunyuan3D-2.1/hy3dpaint/custom_rasterizer",
      message: "uv pip install --no-build-isolation -e ."
    }
  }, {
    method: "shell.run",
    params: {
      conda: { path: "../../../../env", python: "3.10" },
      shell: "{{which('bash')}}",
      path: "app/services/hunyuan3d/vendor/Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer",
      message: "bash compile_mesh_painter.sh"
    }
  }, {
    when: "{{!exists('app/services/hunyuan3d/vendor/Hunyuan3D-2.1/hy3dpaint/ckpt/RealESRGAN_x4plus.pth')}}",
    method: "fs.download",
    params: {
      url: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
      path: "app/services/hunyuan3d/vendor/Hunyuan3D-2.1/hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
    }
  }, {
    method: "fs.write",
    params: {
      path: "app/services/hunyuan3d/env/.maestro_hunyuan3d_v1.installed",
      text: "ok"
    }
  }]
}
