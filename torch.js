module.exports = {
  run: [
    // windows nvidia
    {
      "when": "{{platform === 'win32'}}",
      "method": "shell.run",
      "params": {
        "venv": "{{args && args.venv ? args.venv : null}}",
        "path": "{{args && args.path ? args.path : '.'}}",
        "message": [
          "uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 {{args && args.xformers ? 'xformers==0.0.30' : ''}} --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps",
          "uv pip install triton-windows==3.3.1.post19",
          "uv pip install https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows/sageattention-2.2.0+cu128torch2.7.1-cp310-cp310-win_amd64.whl",
          "uv pip install https://huggingface.co/cocktailpeanut/wheels/resolve/main/flash_attn-2.8.2%2Bcu128torch2.7-cp310-cp310-win_amd64.whl"
        ]
      }
    },
    // linux nvidia
    {
      "when": "{{platform === 'linux'}}",
      "method": "shell.run",
      "params": {
        "venv": "{{args && args.venv ? args.venv : null}}",
        "path": "{{args && args.path ? args.path : '.'}}",
        "message": [
          "uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 {{args && args.xformers ? 'xformers==0.0.30' : ''}} --index-url https://download.pytorch.org/whl/cu128 --force-reinstall",
          "uv pip install https://huggingface.co/MonsterMMORPG/SECourses_Premium_Flash_Attention/resolve/main/sageattention-2.1.1-cp310-cp310-linux_x86_64.whl",
          "uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.7.4+cu128torch2.7-cp310-cp310-linux_x86_64.whl",
          "uv pip install numpy==2.1.2"
        ]
      }
    },
    // Marker file so update.js can skip this script on routine updates
    // (torch + triton + sage + flash already installed, no version bump).
    // Saves ~60-120s of unnecessary re-download every time the user runs
    // Update with nothing new to install.
    //
    // When bumping ANY version above (torch / triton / sage / flash), ALSO
    // bump the `_v1` suffix here AND in update.js's gate to force a
    // reinstall on the next update. The old marker becomes stale and the
    // `!exists(new_marker)` gate evaluates true → this script runs → new
    // marker written. Old marker stays as harmless cruft until reset.js.
    {
      "method": "fs.write",
      "params": {
        "path": "app/env/.maestro_torch_v1.installed",
        "text": "torch + triton + sage + flash installed by torch.js. Delete this file to force update.js to re-run torch.js on the next Update."
      }
    }
  ]
}
