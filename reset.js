// Wipes every install artifact so the next Install starts from scratch.
// Mirrors the directories created by install.js and sam_install.js.
module.exports = {
  run: [
    // Main Python venv
    { method: "fs.rm", params: { path: "app/env" } },
    // SAM 3.1 Python 3.12 conda env
    { method: "fs.rm", params: { path: "app/services/sam/env" } },
    // SAM 3 source checkout (will be re-cloned on install)
    { method: "fs.rm", params: { path: "app/services/sam/sam3" } },
    // Hunyuan3D isolated runtime, official source checkouts, and model cache
    { method: "fs.rm", params: { path: "app/services/hunyuan3d/env" } },
    { method: "fs.rm", params: { path: "app/services/hunyuan3d/vendor" } },
    { method: "fs.rm", params: { path: "app/ckpts/model3d" } },
    // UI build artifacts
    { method: "fs.rm", params: { path: "ui/node_modules" } },
    { method: "fs.rm", params: { path: "ui/dist" } }
  ]
}
